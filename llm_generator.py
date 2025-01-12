import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import warnings
from pdfminer.high_level import extract_text

warnings.filterwarnings("ignore")

load_dotenv()

class ResumeCustomizer:

    def __init__(self):
        print("Resume Customizer Initialized")

    def llm_inference(
        self,
        model_type: str,
        resume_text: str,
        job_description: str,
        context: str,
        prompt_template: str,
        openai_model_name: str = "",
        hf_repo_id: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature: float = 0.1,
        max_length: int = 1024,
    ) -> str:
        """
        Generate a customized resume based on the job description using HuggingFace or OpenAI LLMs.

        Args:
            model_type: 'openai' or 'huggingface'.
            resume_text: The candidate's current resume.
            job_description: The job description for customization.
            prompt_template: The prompt template to instruct the LLM.
            openai_model_name: OpenAI model name (e.g., gpt-3.5-turbo, gpt-4).
            hf_repo_id: HuggingFace model repo ID.
            temperature: Controls randomness. Lower for deterministic outputs.
            max_length: Maximum tokens for the response.

        Returns:
            Customized resume as a string.
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "context",
                "resume_text",
                "job_description"],
        )

        if model_type == "openai":
            llm = OpenAI(
                model_name=openai_model_name, temperature=temperature, max_tokens=max_length
            )
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            return llm_chain.run(
                resume_text=resume_text,
                job_description=job_description,
            )

        elif model_type == "huggingface":
            llm = HuggingFaceHub(
                repo_id=hf_repo_id,
                model_kwargs={"temperature": temperature, "max_length": max_length},
            )
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            response = llm_chain.run(
                context=context,
                resume_text=resume_text,
                job_description=job_description,
            )
            # Extracting only the response part from the output
            response_start_index = response.find('Response;')
            response_output = response[response_start_index + len('Response;'):].strip()
    
            return response_output

        else:
            print("Invalid model_type. Choose either 'openai' or 'huggingface'.")
            return None


if __name__ == "__main__":
    
    # Ensure .env file contains 'HUGGINGFACEHUB_API_TOKEN' and 'OPENAI_API_KEY'
    openai_api_key = os.getenv("OPENAI_API_TOKEN")
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Example Inputs
    # resume_text = """
    # Shantanu Kolekar
    # Data Analyst with 3+ years of experience in data analysis, ETL processes, and dashboard creation using Power BI and Tableau.
    # Skilled in Python, SQL, and AWS. Proven ability to derive actionable insights from complex datasets.
    # """
    
    # job_description = """
    # We are seeking a Data Analyst with expertise in SQL, Python, and Power BI.
    # The ideal candidate should have experience in ETL processes, data visualization, and cloud platforms like AWS.
    # Knowledge of machine learning and predictive analytics is a plus.
    # """


    resume_text = extract_text('resume.pdf')
    job_description = extract_text('jd.pdf')
    context = "You are an expert resume builder."

    # Prompt Template for Resume Customization
    prompt_template = f"""INSTRUCTIONS: {context}

    The following is a candidate's resume:
    {resume_text}

    The following is the job description:
    {job_description}

    Customize the resume in following manner:
    
    Your task is to improve the Skills, Experience and Summary section only to better align with the job description by highlighting relevant skills & experience.

    Don Not Make any changes to Education section.

    Ensure the tone is professional and the formatting is suitable for a resume.
    
    Response;
    """

    # Initialize the Resume Customizer
    model = ResumeCustomizer()

    # Generate Customized Resume using HuggingFace or OpenAI
    customized_resume = model.llm_inference(
        model_type="huggingface",  # Choose 'openai' or 'huggingface'
        resume_text=resume_text,
        job_description=job_description,
        context=context,
        prompt_template=prompt_template,
        openai_model_name="gpt-3.5-turbo",
        hf_repo_id="openai-community/gpt2", # mistralai/Mistral-7B-v0.1
        temperature=0.1,
        max_length=1400,
    )

    # print("Customized Resume:")
    print(customized_resume)
