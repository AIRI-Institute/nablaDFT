def collate_hamiltonian(
    batch, *, default_collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None
):
    pass


def collate_atoms_spk(batch, *, default_collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    idx_keys = {structure.idx_i, structure.idx_j, structure.idx_i_triples}
    # Atom triple indices must be treated separately
    idx_triple_keys = {structure.idx_j_triples, structure.idx_k_triples}

    coll_batch = {}
    for key in elem:
        if (key not in idx_keys) and (key not in idx_triple_keys):
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)
        elif key in idx_keys:
            coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 0)

    seg_m = torch.cumsum(coll_batch[structure.n_atoms], dim=0)
    seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
    idx_m = torch.repeat_interleave(torch.arange(len(batch)), repeats=coll_batch[structure.n_atoms], dim=0)
    coll_batch[structure.idx_m] = idx_m

    for key in idx_keys:
        if key in elem.keys():
            coll_batch[key] = torch.cat([d[key] + off for d, off in zip(batch, seg_m)], 0)

    # Shift the indices for the atom triples
    for key in idx_triple_keys:
        if key in elem.keys():
            indices = []
            offset = 0
            for idx, d in enumerate(batch):
                indices.append(d[key] + offset)
                offset += d[structure.idx_j].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch
