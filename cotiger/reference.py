"""Preserved calibration reference for exact-output and speed comparisons."""
import torch


@torch.inference_mode()
def reference_search(tiger, ids, mask, widths, weights=(1., 1., 1.), sibling=0., penalty_depths=(2,), recorder=None, adjuster=None):
    """Weighted path selection; optional non-cumulative log sibling-rank penalty.

    Zero penalty/unit weights follow the original full-vocabulary Top-K exactly.
    Labels/targets never enter this function. Penalties change selection only;
    true per-token probabilities and cumulative model scores remain separate.
    """
    model = tiger.model
    encoded = model.encoder(input_ids=ids, attention_mask=mask)
    b, k, v = len(ids), widths[0], model.config.vocab_size
    paths = torch.full((b * k, 1), model.config.decoder_start_token_id, device=ids.device, dtype=torch.long)
    raw = torch.zeros(b, k, device=ids.device)
    raw[:, 1:] = -1e9
    weighted = raw.clone()
    components = raw.new_zeros((b, k, 0))
    cache, depth_tokens = None, []
    for step, keep in enumerate(widths):
        first = step == 0 and model.config.first_beam_only
        hidden = encoded[0].unsqueeze(1).expand(-1, k, -1, -1).reshape(b * k, encoded[0].shape[1], -1)
        expanded_mask = mask.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
        out = model(encoder_outputs=encoded if first else (hidden,),
                    attention_mask=mask if first else expanded_mask,
                    decoder_input_ids=paths.view(b, k, -1)[:, 0] if first else paths if cache is None else paths[:, -1:],
                    use_cache=True, past_key_values=cache)
        cache = out.past_key_values
        if first:
            cache = tuple(tuple(None if pair is None else tuple(x.repeat_interleave(k, 0) for x in pair)
                                for pair in layer) for layer in cache)
        lp = out.logits[:, -1].float().log_softmax(-1).view(b, 1 if first else k, v)
        raw_candidates = raw[:, :, None] + lp
        selection = weighted[:, :, None] + weights[step] * lp
        if sibling and step + 1 in penalty_depths:
            ranks = lp.argsort(-1, descending=True).argsort(-1).float()
            selection = selection - sibling * ranks.log1p()
        if adjuster is not None:
            selection = selection + adjuster(step + 1, paths.view(b, k, -1), lp)
        _, flat = selection.reshape(b, -1).topk(keep, -1, sorted=True)
        if recorder is not None:
            recorder(depth=step + 1, batch_size=b, num_beams=k, first_beam_only=first,
                     parent_tokens=paths, parent_scores=raw, next_token_logprobs=lp,
                     selected_flat_indices=flat,
                     selected_cumulative_scores=raw_candidates.reshape(b, -1).gather(1, flat),
                     selection_candidate_scores=selection)
        parent, token = flat // v, flat % v
        rows = torch.arange(b, device=ids.device)[:, None]
        token_lp = lp[rows, torch.zeros_like(parent) if first else parent, token]
        components = torch.cat((components.gather(1, parent[:, :, None].expand(-1, -1, step)), token_lp[:, :, None]), -1)
        raw = raw_candidates.view(b, -1).gather(1, flat)
        # Do not carry sibling penalties forward into model probability scores.
        weighted = weighted.gather(1, parent) + weights[step] * token_lp
        global_parent = (rows * k + parent).reshape(-1)
        paths = torch.cat((paths[global_parent], token.reshape(-1, 1)), -1)
        depth_tokens.append(paths.view(b, keep, -1)[:, :, 1:])
        if step < 2:
            cache = model._reorder_cache(cache, global_parent)
        k = keep
    assert torch.allclose(raw, components.sum(-1), atol=2e-6, rtol=1e-6)
    return paths, components, raw, weighted, depth_tokens
