import hydra
import logging
from omegaconf import DictConfig, OmegaConf



logger=logging.getLogger(__name__)
OmegaConf.register_new_resolver("calc", eval, replace=True)


@hydra.main(version_base=None, config_path="configs", config_name="base-nft")
def main(cfg: DictConfig) -> None:
    # setup_logging()
    # Store the YAML config separately
    OmegaConf.save(cfg, f"{cfg.save_dir}/hparams.yaml")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # print(f"Working directory : {os.getcwd()}")
    # reward = hydra.utils.instantiate(cfg.reward)
    reinl = hydra.utils.instantiate(
        cfg.pipeline,
        model_suite=cfg.model,
        reward=cfg.reward,
        logger=cfg.logger,
        diversity_filter=cfg.diversity_filter,
    )
    reinl.run_rl()


if __name__ == '__main__':
    main()
