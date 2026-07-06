from openai import OpenAI
import os

class LLMProfessionTagger:
    """
    Human career tagger.
    Uses an LLM to assign a profession label to each human and provide initial capability values.
    """
    
    def __init__(self, api_key=None, base_url="https://api.moonshot.cn/v1"):
        """
        Initialize the human profession tagger
        
        Parameters:
            api_key: Moonshot API key; if None, read from environment
            base_url: Moonshot API base URL
        """
        # initialize LLM interface
        self._init_llm_interface(api_key, base_url)
        
        # load profession templates
        self.profession_templates = self._load_profession_templates()
        
    def _init_llm_interface(self, api_key=None, base_url="https://api.moonshot.cn/v1"):
        """
        Initialize LLM interface
        
        Parameters:
            api_key: Moonshot API key; if None, read from environment
            base_url: Moonshot API base URL
        """
        # read API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("MOONSHOT_API_KEY")
            if api_key is None:
                raise ValueError("Moonshot API key not provided and MOONSHOT_API_KEY is not set in environment")
        
        # initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # initialize message history
        self.messages = [
            {
                "role": "system", 
                "content": (
                    "You are a professional human career analysis assistant. "
                    "You classify human descriptions into the best-matching profession type. "
                    "You provide safe, helpful, and accurate answers."
                )
            }
        ]
        
    def _load_profession_templates(self):
        """Load profession templates."""
        # can be loaded from file or database
        return {
            "Operator": {
                "transport": 0.7,
                "coverage": 0.6,
                "precision": 0.8
            },
            "Monitor": {
                "transport": 0.5,
                "coverage": 0.9,
                "precision": 0.7
            },
            "Coordinator": {
                "transport": 0.6,
                "coverage": 0.7,
                "precision": 0.9
            },
            "Expert": {
                "transport": 0.4,
                "coverage": 0.8,
                "precision": 0.95
            },
            "Novice": {
                "transport": 0.5,
                "coverage": 0.5,
                "precision": 0.5
            }
        }
        
    def tag_human_profession(self, human_description):
        """Use LLM to tag a human with a profession."""
        # build detailed prompt
        prompt = f"""
        Based on the following human description, classify them into the best-matching profession type.
        
        Human description: {human_description}
        
        Available profession types:
        - Operator: skilled at operation and control, with strong precision control capability
        - Monitor: skilled at observation and monitoring, with strong coverage control capability
        - Coordinator: skilled at coordination and communication, with strong coordination capability
        - Expert: deep knowledge and extensive experience in a specific domain
        - Novice: limited experience, baseline capability across all areas
        
        Return only the profession name with no other explanation. If the description does not match any type exactly, choose the closest one.
        """
        
        # call LLM
        response = self.generate(prompt)
        print("[DEBUGING] Now we using LLM Tagger to tag the human profession: ")
        print(f"human description: {human_description}")
        print(f"response: {response}")

        profession = response.strip()
        
        # if returned profession is not in templates, use default or extract match
        if profession not in self.profession_templates:
            # try to extract profession name from response
            for template_profession in self.profession_templates.keys():
                if template_profession in profession:
                    profession = template_profession
                    break
            else:
                profession = "Operator"  # default profession
            
        return profession
        
    def get_initial_capabilities(self, profession):
        """Get initial capability values for a profession."""
        return self.profession_templates.get(profession, self.profession_templates["Operator"])
        
    def generate(self, prompt):
        """
        Generate LLM response
        
        Parameters:
            prompt: prompt text
            
        Returns:
            LLM-generated response text
        """
        self.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # call API
        completion = self.client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=self.messages,
            temperature=0.3,
        )
        
        # Get Response
        assistant_message = completion.choices[0].message
        self.messages.append(assistant_message) # Append the response to the history
        
        return assistant_message.content
        
    def reset_conversation(self):
        """Reset conversation history."""
        self.messages = [
            {
                "role": "system", 
                "content": (
                    "You are a professional human career analysis assistant. "
                    "You classify human descriptions into the best-matching profession type. "
                    "You provide safe, helpful, and accurate answers."
                )
            }
        ]
