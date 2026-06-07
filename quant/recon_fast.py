import torch
import torch.nn.functional as F
import copy
from .quant_layer import QuantModule, lp_loss
from .quant_block import BaseQuantBlock
from .adaptive_rounding import AdaRoundQuantizer
from .set_weight_quantize_params import get_init, get_dc_fp_init
from .set_act_quantize_params import set_act_quantize_params


def compute_local_infonce(q_feat, fp_feat, tau=0.1):
    """Compute InfoNCE loss in the local feature space."""
    B = q_feat.shape[0]
    v_q = F.normalize(q_feat.view(B, -1), dim=1)
    v_f = F.normalize(fp_feat.view(B, -1), dim=1)
    sim = torch.mm(v_q, v_f.t()) / tau
    labels = torch.arange(B, device=q_feat.device)
    return F.cross_entropy(sim, labels)


class FastLossFunction:
    """Local loss without the FP tail or prediction-difference term."""

    def __init__(self, module, weight=1., p=2.0, max_count=20000, b_range=(20, 2), warmup=0.2):
        self.module = module
        self.weight = weight
        self.p = p
        self.loss_start = max_count * warmup
        self.count = 0
        self.start_b, self.end_b = b_range[0], b_range[1]
        self.t_max = max_count
        self.start_decay = warmup * max_count

    def get_b(self):
        if self.count < self.start_decay:
            return self.start_b
        else:
            rel_t = (self.count - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

    def __call__(self, pred, tgt):
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)
        b = self.get_b()
        round_loss = 0
        if self.count >= self.loss_start:
            for name, mod in self.module.named_modules():
                if isinstance(mod, QuantModule):
                    round_vals = mod.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        return rec_loss, round_loss, b


def fast_reconstruction(model, fp_model, module, fp_module, cali_data, batch_size=32, iters=20000,
                        weight=0.01, lr=4e-5, b_range=(20, 2), warmup=0.2, input_prob=0.5,
                        keep_gpu=True, use_infonce=True, infonce_lambda=1.0, infonce_tau=0.1):
    """Local reconstruction for layers and blocks without an FP tail."""

    cached_inps = get_init(model, module, cali_data, batch_size=batch_size, input_prob=True, keep_gpu=keep_gpu)
    cached_outs, _, cur_syms = get_dc_fp_init(fp_model, fp_module, cali_data, batch_size=batch_size, input_prob=True,
                                              keep_gpu=keep_gpu)
    set_act_quantize_params(module, cali_data=cached_inps[:min(256, cached_inps.size(0))])

    module.set_quant_state(True, True)
    for para in model.parameters():
        para.requires_grad = False

    w_para, a_para = [], []
    for mod in module.modules():
        if isinstance(mod, QuantModule):
            mod.weight_quantizer = AdaRoundQuantizer(uaq=mod.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                     weight_tensor=mod.org_weight.data)
            mod.weight_quantizer.soft_targets = True
            w_para.append(mod.weight_quantizer.alpha)
        if isinstance(mod, (QuantModule, BaseQuantBlock)):
            if mod.act_quantizer.delta is not None:
                mod.act_quantizer.delta = torch.nn.Parameter(torch.tensor(mod.act_quantizer.delta))
                a_para.append(mod.act_quantizer.delta)
            mod.act_quantizer.is_training = True

    w_opt = torch.optim.Adam(w_para, lr=3e-3) if w_para else None
    a_opt = torch.optim.Adam(a_para, lr=lr) if a_para else None
    a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.) if a_opt else None

    loss_func = FastLossFunction(module, weight=weight, max_count=iters, b_range=b_range, warmup=warmup)
    device = 'cuda'
    sz = cached_inps.size(0)

    try:
        from thop import profile
        import builtins
        with torch.no_grad():
            dummy_inp = cached_inps[0:1].to(device)
            temp_mod = copy.deepcopy(fp_module)
            macs_module, _ = profile(temp_mod, inputs=(dummy_inp,), verbose=False)
            del temp_mod

            flops_per_sample_per_iter = (macs_module * 3) * 2
            if hasattr(builtins, 'GLOBAL_CALIBRATION_FLOPS'):
                builtins.GLOBAL_CALIBRATION_FLOPS += flops_per_sample_per_iter * batch_size * iters
    except Exception as e:
        print(f"[Warning] Skipping FLOPs accounting due to: {e}")

    for i in range(iters):
        idx = torch.randint(0, sz, (batch_size,))
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)

        if w_opt: w_opt.zero_grad()
        if a_opt: a_opt.zero_grad()

        # Concatenate a doubled batch only when input dropping is enabled.
        if input_prob < 1.0:
            cur_sym = cur_syms[idx].to(device)
            drop_inp = torch.where(torch.rand_like(cur_inp) < input_prob, cur_inp, cur_sym)
            cur_inp_cat = torch.cat((drop_inp, cur_inp))
            out_all = module(cur_inp_cat)
            out_drop = out_all[:batch_size]
            out_quant = out_all[batch_size:]
        else:
            out_drop = module(cur_inp)
            out_quant = out_drop

        rec_loss, round_loss, b = loss_func(out_drop, cur_out)

        loss_infonce = 0.0
        if use_infonce:
            loss_infonce = compute_local_infonce(out_quant, cur_out, tau=infonce_tau)
            total_loss = rec_loss + round_loss + infonce_lambda * loss_infonce
        else:
            total_loss = rec_loss + round_loss

        if loss_func.count % 500 == 0:
            round_val = round_loss.item() if not isinstance(round_loss, int) else 0.0
            info_val = loss_infonce.item() if use_infonce else 0.0
            print(
                f'Iter {i:05d} | Total: {total_loss.item():.4f} | MSE: {rec_loss.item():.4f} | InfoNCE: {info_val:.4f} | Round: {round_val:.4f} | b={b:.2f}')

        # Avoid retaining the graph to keep memory usage bounded.
        total_loss.backward()

        if w_opt: w_opt.step()
        if a_opt: a_opt.step()
        if a_scheduler: a_scheduler.step()

    torch.cuda.empty_cache()
    for mod in module.modules():
        if isinstance(mod, QuantModule):
            mod.weight_quantizer.soft_targets = False
        if isinstance(mod, (QuantModule, BaseQuantBlock)):
            mod.act_quantizer.is_training = False
            mod.trained = True
    for mod in fp_module.modules():
        if isinstance(mod, (QuantModule, BaseQuantBlock)):
            mod.trained = True
