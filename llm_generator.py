import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import warnings
from pdfminer.high_level import extract_text

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

class ResumeCustomizer:
    def __init__(self):
        print("Resume Customizer Initialized")

    def llm_inference_stream(
        self,
        model_type: str,
        resume_text: str,
        job_description: str,
        context: str,
        prompt_template: str,
        openai_model_name: str = "gpt-3.5-turbo",
        openai_api_key: str = "",
        hf_repo_id: str = "mistralai/Mistral-7B-Instruct-v0.1",
        temperature: float = 0.1,
    ) -> str:
        """
        Generate a customized resume based on the job description using HuggingFace or OpenAI LLMs with streaming.

        Args:
            model_type: 'openai' or 'huggingface'.
            resume_text: The candidate's current resume.
            job_description: The job description for customization.
            prompt_template: The prompt template to instruct the LLM.
            openai_model_name: OpenAI model name (e.g., gpt-3.5-turbo, gpt-4).
            hf_repo_id: HuggingFace model repo ID.
            temperature: Controls randomness. Lower for deterministic outputs.

        Returns:
            Customized resume as a string.
        """
        if model_type == "openai":
            from openai import OpenAI
            
            client = OpenAI(api_key=openai_api_key)
            try:
                response = client.chat.completions.create(
                    model=openai_model_name,
                    messages=[
                        {"role": "system", "content": context},
                        {"role": "user", "content": prompt_template},
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content

            except Exception as e:
                print(f"Error during OpenAI API call: {e}")
                return None


        elif model_type == "huggingface":
            # HuggingFace API (streaming not supported)
            llm = HuggingFaceHub(
                repo_id=hf_repo_id,
                model_kwargs={"temperature": temperature, "max_length": 1024},
            )
            llm_chain = LLMChain(prompt=prompt_template, llm=llm)
            response = llm_chain.run(
                context=context,
                resume_text=resume_text,
                job_description=job_description,
            )
            return response

        else:
            print("Invalid model_type. Choose either 'openai' or 'huggingface'.")
            return None


if __name__ == "__main__":
    # Ensure .env file contains 'HUGGINGFACEHUB_API_TOKEN' and 'OPENAI_API_KEY'
    openai_api_token = os.getenv("OPENAI_API_KEY")
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    resume_text = extract_text('resume.pdf')
    job_description = extract_text('jd.pdf')

    context = "You are an expert resume builder."

    # Prompt Template for Resume Customization
    prompt_template = f"""
    INSTRUCTIONS: {context}

    The following is a candidate's resume:
    {resume_text}

    The following is the job description:
    {job_description}

    Customize the resume in the following manner:
    - Improve the SKILLS, EXPERIENCE, and SUMMARY sections to better align with the job description.
    - Highlight relevant skills and experience for the role.
    - Maintain a professional tone and proper formatting.
    """

    # Initialize the Resume Customizer
    model = ResumeCustomizer()

    # Generate Customized Resume using OpenAI with Streaming
    customized_resume = model.llm_inference_stream(
        model_type="openai",  # Choose 'openai' or 'huggingface'
        openai_api_key = openai_api_token,
        resume_text=resume_text,
        job_description=job_description,
        context=context,
        prompt_template=prompt_template,
        openai_model_name="gpt-4",
        temperature=0.7,
    )

    # Print the final customized resume
    print("\n\nFinal Customized Resume:\n")
    print(customized_resume)
