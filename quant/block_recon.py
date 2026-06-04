import torch
import torch.nn.functional as F
from .quant_layer import QuantModule, lp_loss
from .quant_model import QuantModel
from .quant_block import BaseQuantBlock, specials_unquantized
from .adaptive_rounding import AdaRoundQuantizer
from .set_weight_quantize_params import get_init, get_dc_fp_init
from .set_act_quantize_params import set_act_quantize_params

# ================= Global InfoNCE module =================
try:
    from .infonce_loss import InfoNCELoss
except ImportError:
    pass


def _build_fp_tail_model(fp_model, fp_block, cali_data, batch_size, device):
    """Build the FP tail mapper g = f[k:]_F for the contrastive space."""

    class TailModelWrapper(torch.nn.Module):
        def __init__(self, fp_model, fp_block):
            super().__init__()
            self.fp_model = fp_model
            self.fp_block = fp_block
            self._dummy_input = None
            for p in self.fp_model.parameters():
                p.requires_grad_(False)
            self.fp_model.eval()

        def set_dummy_input(self, x):
            self._dummy_input = x.detach()

        def forward(self, act: torch.Tensor) -> torch.Tensor:
            B = act.shape[0]
            dummy = self._dummy_input[:B].to(act.device)
            hooked = [False]

            def replace_hook(mod, inp, out):
                if not hooked[0]:
                    hooked[0] = True
                    return act
                return out

            handle = self.fp_block.register_forward_hook(replace_hook)
            try:
                if act.requires_grad:
                    result = self.fp_model(dummy)
                else:
                    with torch.no_grad():
                        result = self.fp_model(dummy)
            finally:
                handle.remove()
            return result

    wrapper = TailModelWrapper(fp_model, fp_block)
    dummy = cali_data[:min(batch_size, cali_data.shape[0])].to(device)
    wrapper.set_dummy_input(dummy)
    return wrapper


# ===================================================

include = False


def find_unquantized_module(model: torch.nn.Module, module_list: list = [], name_list: list = []):
    """Store subsequent unquantized modules in a list"""
    global include
    for name, module in model.named_children():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if not module.trained:
                include = True
                module.set_quant_state(False, False)
                name_list.append(name)
                module_list.append(module)
        elif include and type(module) in specials_unquantized:
            name_list.append(name)
            module_list.append(module)
        else:
            find_unquantized_module(module, module_list, name_list)
    return module_list[1:], name_list[1:]


def block_reconstruction(model: QuantModel, fp_model: QuantModel, block: BaseQuantBlock, fp_block: BaseQuantBlock,
                         cali_data: torch.Tensor, batch_size: int = 32, iters: int = 20000, weight: float = 0.01,
                         opt_mode: str = 'mse', b_range: tuple = (20, 2),
                         warmup: float = 0.0, p: float = 2.0, lr: float = 4e-5,
                         input_prob: float = 1.0, keep_gpu: bool = True,
                         lamb_r: float = 0.2, T: float = 7.0, bn_lr: float = 1e-3, lamb_c=0.02,
                         # Global InfoNCE options.
                         use_infonce=False, infonce_lambda=1.0, infonce_tau=0.1):
    """
    Reconstruction to optimize the output from each block.
    """

    '''get input and set scale'''
    cached_inps = get_init(model, block, cali_data, batch_size=batch_size,
                           input_prob=True, keep_gpu=keep_gpu)
    cached_outs, cached_output, cur_syms = get_dc_fp_init(fp_model, fp_block, cali_data, batch_size=batch_size,
                                                          input_prob=True, keep_gpu=keep_gpu, bn_lr=bn_lr, lamb=lamb_c)
    set_act_quantize_params(block, cali_data=cached_inps[:min(256, cached_inps.size(0))])

    '''set state'''
    cur_weight, cur_act = True, True

    global include
    module_list, name_list, include = [], [], False
    module_list, name_list = find_unquantized_module(model, module_list, name_list)
    block.set_quant_state(cur_weight, cur_act)
    for para in model.parameters():
        para.requires_grad = False

    '''set quantizer'''
    round_mode = 'learned_hard_sigmoid'
    # Replace weight quantizer to AdaRoundQuantizer
    w_para, a_para = [], []
    w_opt, a_opt = None, None
    scheduler, a_scheduler = None, None

    for module in block.modules():
        '''weight'''
        if isinstance(module, QuantModule):
            module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                        weight_tensor=module.org_weight.data)
            module.weight_quantizer.soft_targets = True
            w_para += [module.weight_quantizer.alpha]
        '''activation'''
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if module.act_quantizer.delta is not None:
                module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                a_para += [module.act_quantizer.delta]
            '''set up drop'''
            module.act_quantizer.is_training = True

    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=3e-3)
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.)

    loss_mode = 'relaxation'
    rec_loss = opt_mode
    loss_func = LossFunction(block, round_loss=loss_mode, weight=weight, max_count=iters, rec_loss=rec_loss,
                             b_range=b_range, decay_start=0, warmup=warmup, p=p, lam=lamb_r, T=T)
    device = 'cuda'
    sz = cached_inps.size(0)

    # Initialize the contrastive loss and precompute FP features.
    infonce_loss_fn = None
    cached_v_f = None  # Static global memory bank of precomputed FP features.
    fp_tail = None
    if use_infonce:
        fp_tail = _build_fp_tail_model(fp_model, fp_block, cali_data, batch_size, device)
        infonce_loss_fn = InfoNCELoss(fp_tail, tau=infonce_tau)

        print("Pre-computing FP contrastive features for Block (Global Memory Bank)...")
        with torch.no_grad():
            v_f_list = []
            for start_idx in range(0, sz, batch_size):
                end_idx = min(start_idx + batch_size, sz)
                batch_outs = cached_outs[start_idx:end_idx].to(device)
                v_f_batch = infonce_loss_fn._map_features(batch_outs)
                v_f_list.append(v_f_batch.cpu())
            cached_v_f = torch.cat(v_f_list, dim=0).to(device)

    # Account for the current block and FP tail FLOPs.
    try:
        from thop import profile
        import builtins
        import copy
        with torch.no_grad():
            dummy_inp = cached_inps[0:1].to(device)
            # Use deepcopy to avoid mutating the original module during profiling.
            temp_block = copy.deepcopy(block)
            macs_module, _ = profile(temp_block, inputs=(dummy_inp,), verbose=False)
            del temp_block

            # Profile the FP tail when contrastive calibration is enabled.
            macs_tail = 0
            if use_infonce and fp_tail is not None:
                dummy_out = block(dummy_inp)
                temp_tail = copy.deepcopy(fp_tail)
                macs_tail, _ = profile(temp_tail, inputs=(dummy_out,), verbose=False)
                del temp_tail

            # Estimate total calibration FLOPs for this stage.
            flops_per_sample_per_iter = (macs_module * 3 + macs_tail) * 2
            total_flops_this_stage = flops_per_sample_per_iter * batch_size * iters

            if hasattr(builtins, 'GLOBAL_CALIBRATION_FLOPS'):
                builtins.GLOBAL_CALIBRATION_FLOPS += total_flops_this_stage
    except Exception as e:
        print(f"[Warning] Skipping FLOPs accounting due to: {e}")

    for i in range(iters):
        idx = torch.randint(0, sz, (batch_size,))
        cur_inp = cached_inps[idx].to(device)
        cur_sym = cur_syms[idx].to(device)
        output_fp = cached_output[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        if input_prob < 1.0:
            drop_inp = torch.where(torch.rand_like(cur_inp) < input_prob, cur_inp, cur_sym)

        cur_inp = torch.cat((drop_inp, cur_inp))

        if w_opt:
            w_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()

        out_all = block(cur_inp)

        '''forward for prediction difference'''
        out_drop = out_all[:batch_size]
        out_quant = out_all[batch_size:]
        output = out_quant
        for num, module in enumerate(module_list):
            # for ResNet and RegNet
            if name_list[num] == 'fc':
                output = torch.flatten(output, 1)
            # for MobileNet and MNasNet
            if isinstance(module, torch.nn.Dropout):
                output = output.mean([2, 3])
            output = module(output)

        err = loss_func(out_drop, cur_out, output, output_fp)

        # Add global InfoNCE to the reconstruction loss when enabled.
        if use_infonce and infonce_loss_fn is not None:
            infonce_loss_val = infonce_lambda * infonce_loss_fn(out_quant, cached_v_f, idx.to(device))
            err = err + infonce_loss_val

            if loss_func.count % 500 == 0 or loss_func.count == 1:
                print(f"      ---> [Global InfoNCE] Scaled loss: {infonce_loss_val.item():.4f} | Base loss: {(err - infonce_loss_val).item():.4f}")

        err.backward(retain_graph=True)
        if w_opt:
            w_opt.step()
        if a_opt:
            a_opt.step()
        if scheduler:
            scheduler.step()
        if a_scheduler:
            a_scheduler.step()
    torch.cuda.empty_cache()

    for module in block.modules():
        if isinstance(module, QuantModule):
            '''weight '''
            module.weight_quantizer.soft_targets = False
        '''activation'''
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.act_quantizer.is_training = False
            module.trained = True
    for module in fp_block.modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.trained = True


class LossFunction:
    def __init__(self,
                 block: BaseQuantBlock,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 lam: float = 1.0,
                 T: float = 7.0):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.lam = lam
        self.T = T

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.pd_loss = torch.nn.KLDivLoss(reduction='batchmean')

    def __call__(self, pred, tgt, output, output_fp):
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        pd_loss = self.pd_loss(F.log_softmax(output / self.T, dim=1), F.softmax(output_fp / self.T, dim=1)) / self.lam

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss + pd_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, pd:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(pd_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
