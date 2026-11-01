import csv
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


folder = Path(__file__).parent
laptop_file = folder / "products.csv"

with laptop_file.open(newline="") as file:
    LAPTOPS = list(csv.DictReader(file))

with (folder / "applications.csv").open(newline="") as file:
    APPLICATIONS = list(csv.DictReader(file))

for laptop in LAPTOPS:
    laptop["price"] = float(laptop["price"])
    laptop["stock"] = int(laptop["stock"])


def find_application(name):
    for application in APPLICATIONS:
        if application["application_name"].lower() == name.strip().lower():
            return application
    return None


purchase_done = False


@tool
def get_application_requirement(application_name: str) -> dict:
    """Look up an application's required processor. Return supported apps if unknown."""
    application = find_application(application_name)
    if application is None:
        return {"error": "Unknown application", "supported_applications": APPLICATIONS}
    return application


@tool
def search_laptops(processor_type: str, budget: float) -> dict:
    """Find available laptops matching the processor and budget, cheapest first."""
    matches = []
    for laptop in LAPTOPS:
        same_processor = laptop["processor type"].lower() == processor_type.strip().lower()
        if same_processor and laptop["stock"] > 0 and laptop["price"] <= budget:
            matches.append(laptop)

    matches.sort(key=lambda laptop: laptop["price"])
    return {
        "processor_type": processor_type,
        "budget": budget,
        "matching_laptops": matches,
        "message": "Cheapest first" if matches else "No matching laptop within budget. Do not buy.",
    }


@tool
def buy_laptop(laptop_id: str, application_name: str, budget: float) -> dict:
    """Simulate buying one laptop when requested. Check app, price, and stock.
    Update the laptop list only; no real payment or order is placed.
    """
    global purchase_done
    if purchase_done:
        return {"error": "Already purchased a laptop for this request."}

    application = find_application(application_name)
    if application is None:
        return {"error": "Unknown application."}

    for laptop in LAPTOPS:
        if laptop["id"] == laptop_id:
            if laptop["processor type"] != application["needed_processor_type"]:
                return {"error": "Processor does not match the application."}
            if not laptop["price"] <= budget:
                return {"error": "Laptop exceeds budget."}
            if laptop["stock"] <= 0:
                return {"error": "Laptop is out of stock."}

            laptop["stock"] -= 1
            purchase_done = True
            return {
                "status": "SIMULATED PURCHASE SUCCESSFUL",
                "name": laptop["name"],
                "processor_type": laptop["processor type"],
                "application": application_name,
                "price": laptop["price"],
                "remaining_stock": laptop["stock"],
            }

    return {"error": "Unknown laptop ID."}


def save_laptops():
    with laptop_file.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["id", "name", "processor type", "price", "stock"]
        )
        writer.writeheader()
        writer.writerows(LAPTOPS)


model = ChatOllama(
    model="qwen3:1.7b",
    base_url="http://127.0.0.1:11434",
    temperature=0,
    reasoning=True,
    num_ctx=4096,
    num_predict=2048,
)

SYSTEM_PROMPT = """You help users buy one laptop from a fictional local store.
Read the conversation to identify the application and the latest maximum budget
provided by the user. If either is missing, ask a question to get it.
Do not ask again for an application already named in the conversation.
Never ask the user for processor requirements: get_application_requirement supplies them.
Missing budget: ask "What is your maximum budget?"
Missing application: ask "Which application do you want to run?"
Do not call tools until the user has supplied both. Never assume a budget.
Do not print field labels like "Budget: Missing". Ask the user instead.

Once you have both pieces of information, use tools to complete the task:
Call get_application_requirement for the application.
Call search_laptops with the returned processor type and the user's budget.
If no match exists, tell the user and wait for a revised budget.
If a match exists and the user wants to buy, call buy_laptop for the first match.
You must actually call buy_laptop; saying you will buy does not buy anything.
If the user only wants a recommendation or says not to buy, do not buy.
If the application is unknown, ask the user to choose a supported application.
Use exact product IDs and processor types from tools. Prices include all costs.
Buy only one laptop per task. A follow up budget continues the previous task.
After success, do not buy again unless the user asks for another laptop.
Only claim purchase success when the tool confirms it. Say it is simulated.
Do not invent prices or requirements. Keep the final answer short."""

agent = create_agent(
    model=model,
    tools=[get_application_requirement, search_laptops, buy_laptop],
    system_prompt=SYSTEM_PROMPT,
)

conversation = []


def run_request(request):
    global purchase_done
    purchase_done = False
    conversation.append({"role": "user", "content": request})

    for update in agent.stream(
        {"messages": conversation.copy()},
        config={"recursion_limit": 20},
        stream_mode="updates",
    ):
        for step in update.values():
            messages = step.get("messages", [])
            conversation.extend(messages)
            for message in messages:
                if message.type == "tool":
                    print("TOOL RESULT:", message.content)
                elif message.type == "ai":
                    for call in message.tool_calls:
                        print("TOOL CALL:", call["name"], call["args"])
                    if not message.tool_calls and message.content:
                        print("ASSISTANT:", message.content)


if __name__ == "__main__":
    print("Laptop Shopping Agent")
    print("Simulated purchases. Type quit, exit, or /bye to save stock and leave.")

    while True:
        request = input("\nYour request: ").strip()
        if request.lower() in {"quit", "exit", "/bye"}:
            save_laptops()
            print("Stock saved.")
            break
        if request:
            run_request(request)
