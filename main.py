import asyncio

from src.agent.dispatch import DispatchAgent
from src.config.settings import Settings


async def main():
    settings = Settings()
    agent = DispatchAgent(settings)

    response = await agent.handle_voice_input("tech-001")
    print(f"Response: {response}")

    stats = agent.get_statistics()
    print(f"Stats: {stats}")



if __name__ == "__main__":
    asyncio.run(main())