import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv()

class Solver:
    def __init__(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name='gemini-1.5-pro-latest')
    
    def solve(self, image_path: str):

        image = Image.open(image_path)
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        response = self.model.generate_content([os.environ.get("SYSTEM_PROMPT"),image])
        query_response = response.text
        print(query_response)
        return query_response
    

if __name__ == "__main__":
    solver = Solver()
    print(solver.solve())