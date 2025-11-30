from openai import OpenAI
import os

class LLMProfessionTagger:
    """
    Human Career打标器
    使用LLM为人类打标职业，并提供职业的初始能力值
    """
    
    def __init__(self, api_key=None, base_url="https://api.moonshot.cn/v1"):
        """
        初始化人类职业打标器
        
        参数:
            api_key: Moonshot API密钥，如果为None则从环境变量获取
            base_url: Moonshot API基础URL
        """
        # 初始化LLM接口
        self._init_llm_interface(api_key, base_url)
        
        # 加载职业模板
        self.profession_templates = self._load_profession_templates()
        
    def _init_llm_interface(self, api_key=None, base_url="https://api.moonshot.cn/v1"):
        """
        初始化LLM接口
        
        参数:
            api_key: Moonshot API密钥，如果为None则从环境变量获取
            base_url: Moonshot API基础URL
        """
        # 如果未提供API密钥，则从环境变量获取
        if api_key is None:
            api_key = os.environ.get("MOONSHOT_API_KEY")
            if api_key is None:
                raise ValueError("未提供Moonshot API密钥，且环境变量中未设置MOONSHOT_API_KEY")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 初始化消息历史
        self.messages = [
            {
                "role": "system", 
                "content": "你是一个专业的人类职业分析助手，擅长根据人类描述将其归类为最匹配的职业类型。你会提供安全、有帮助、准确的回答。"
            }
        ]
        
    def _load_profession_templates(self):
        """加载职业模板"""
        # 可以从文件或数据库加载
        return {
            "操作员": {
                "transport": 0.7,
                "coverage": 0.6,
                "precision": 0.8
            },
            "监控员": {
                "transport": 0.5,
                "coverage": 0.9,
                "precision": 0.7
            },
            "协调员": {
                "transport": 0.6,
                "coverage": 0.7,
                "precision": 0.9
            },
            "专家": {
                "transport": 0.4,
                "coverage": 0.8,
                "precision": 0.95
            },
            "新手": {
                "transport": 0.5,
                "coverage": 0.5,
                "precision": 0.5
            }
        }
        
    def tag_human_profession(self, human_description):
        """使用LLM为人类打标职业"""
        # 构建更详细的提示
        prompt = f"""
        请根据以下人类描述，将其归类为最匹配的职业类型。
        
        人类描述: {human_description}
        
        可选职业类型:
        - 操作员: 擅长操作和控制，具有较高的精确控制能力
        - 监控员: 擅长观察和监控，具有较高的覆盖控制能力
        - 协调员: 擅长协调和沟通，具有较高的协调能力
        - 专家: 在特定领域具有深厚知识和丰富经验
        - 新手: 经验较少，各项能力处于基础水平
        
        请只返回职业名称，不要有其他解释。如果描述的职业不在上述列表中，请选择最接近的职业。
        """
        
        # 调用LLM生成响应
        response = self.generate(prompt)
        print("[DEBUGING] Now we using LLM Tagger to tag the human profession: ")
        print(f"human description: {human_description}")
        print(f"response: {response}")

        profession = response.strip()
        
        # 如果返回的职业不在模板中，使用默认值
        if profession not in self.profession_templates:
            # 尝试从响应中提取职业名称
            for template_profession in self.profession_templates.keys():
                if template_profession in profession:
                    profession = template_profession
                    break
            else:
                profession = "操作员"  # 默认职业
            
        return profession
        
    def get_initial_capabilities(self, profession):
        """获取职业的初始能力值"""
        return self.profession_templates.get(profession, self.profession_templates["操作员"])
        
    def generate(self, prompt):
        """
        生成LLM响应
        
        参数:
            prompt: 提示文本
            
        返回:
            LLM生成的响应文本
        """
        self.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # 调用API
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
        """重置对话历史"""
        self.messages = [
            {
                "role": "system", 
                "content": "你是一个专业的人类职业分析助手，擅长根据人类描述将其归类为最匹配的职业类型。你会提供安全、有帮助、准确的回答。"
            }
        ]