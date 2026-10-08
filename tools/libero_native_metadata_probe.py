"""Shared strict metadata-only instantiation; never alter training config."""
def instantiate_metadata_dataset(config, *, omega_conf, instantiate):
    probe=omega_conf.create(omega_conf.to_container(config,resolve=True))
    target=probe.resolver.get('_target_')
    if target!='egomimic.rldb.zarr.libero_dataset.LiberoReplayResolver':
        raise ValueError('metadata probe requires exact native LIBERO resolver')
    probe.resolver.decoded_cache=False
    dataset=instantiate(probe)
    if dataset.resolver.decoded_cache or dataset.resolver._decoded is not None:
        raise ValueError('metadata probe must not materialize decoded replay')
    return dataset
