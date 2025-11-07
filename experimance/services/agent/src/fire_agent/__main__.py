"""CLI for Fire Agent Service."""
from experimance_common.cli import create_simple_main
from .config import FireAgentServiceConfig, DEFAULT_CONFIG_PATH
from .service import FireAgentService
from agent import SERVICE_TYPE

async def run_service(config_path=None, args=None):
    config = FireAgentServiceConfig.from_overrides(config_file=config_path, args=args)
    service = FireAgentService(config=config)
    await service.start()
    await service.run()


main = create_simple_main(
    service_name="Fire Agent",
    service_type=SERVICE_TYPE,
    service_runner=run_service,
    default_config_path=DEFAULT_CONFIG_PATH,
    config_class=FireAgentServiceConfig,
)

if __name__ == "__main__":
    main()
