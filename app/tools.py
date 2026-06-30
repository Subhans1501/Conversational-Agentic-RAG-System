import datetime
import math
import re

class AgentTools:
    @staticmethod
    def calculate(expression: str) -> str:
        try:

            safe_expr = re.sub(r'[^0-9+\-*/().]', '', expression)
            
            if not safe_expr:
                return "Error: No valid mathematical expression found."

            result = eval(safe_expr, {"__builtins__": None}, math.__dict__)
            return f"The mathematical result of {safe_expr} is {result}."
        except Exception as e:
            return f"Error: Could not calculate the expression. ({str(e)})"
    @staticmethod
    def get_current_time() -> str:
        now = datetime.datetime.now()
        formatted_time = now.strftime("%A, %B %d, %Y at %I:%M %p")
        return f"The current live system time is {formatted_time}."

    @staticmethod
    def get_weather(location: str) -> str:
        """
        Simulates an external Weather API call.
        (Mocked for FYP presentation stability, preventing API key failures).
        """
        location = location.strip().title()
        
        # A simple mock database for demonstration
        mock_database = {
            "Lahore": "Sunny and 34°C with a slight breeze.",
            "Karachi": "Humid and 31°C.",
            "Islamabad": "Pleasant and 26°C, partly cloudy.",
            "London": "Raining and 12°C."
        }
        
        # Default fallback if the city isn't in our mock DB
        weather = mock_database.get(location, f"Clear skies and 24°C.")
        return f"The current weather in {location} is: {weather}"
if __name__ == "__main__":
    print("Testing External Tools...\n")
    tools = AgentTools()
    print("1. Testing Time:")
    print(tools.get_current_time())
    print("\n2. Testing Calculator (150 * 4.5):")
    print(tools.calculate("150 * 4.5"))
    print("\n3. Testing Weather (Lahore):")
    print(tools.get_weather("Lahore"))