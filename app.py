import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Define function to get LLM instance
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

# Define function to generate system prompt
def get_system_prompt(desc):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert in creating clear, concise, and purpose-driven prompts for professionals.  
        Write a system prompt that instructs a specialist to complete a specific task based on a given query.  
        The query will be provided as input.  

        The system prompt should clearly specify the specialist's role and provide detailed instructions to complete the task effectively, aligning with the context and requirements of the query.  
        Include all necessary details to ensure the task is completed accurately, efficiently, and in line with the query's intent.  

        Query: {query}  

        You can start with:  
        'You are an Expert [Content type or domain-specific] specialist...', 'You are a [Content type or domain-specific] professional with 20 years of experience...', 'You are a tech-savvy enthusiast skilled in [Content type/task domain]...'  
        Don't give a header or footer. Never start with 'Task Completion Prompt...'. No emojis. No bold.  

        Identify the domain or context specific to the query and specify the role clearly at the beginning of the prompt.  
        Avoid generic roles; instead, use specific ones such as 'Healthcare Operations Specialist,' 'Technology Product Strategist,' 'Finance Data Analyst,' etc. 
        """
    )
    chain = prompt | llm
    system_prompt = chain.invoke(desc).content
    return system_prompt

def get_human_prompt(desc):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert in creating clear, concise, and actionable instructions for humans.  
        Write a human prompt that provides a clear and specific task to be completed based on a given query.  
        The query will be provided as input.  

        The human prompt should be clear, task-focused, and easy to understand.  
        Include necessary details, but avoid overloading with information.  

        Query: {query}  

        Example start: 'Please write...', 'Your task is to...', 'Provide a...'  

        Avoid abstract or overly general instructions. Make the task actionable, with clear deliverables or objectives.
        """
    )
    chain = prompt | llm
    human_prompt = chain.invoke(desc).content
    return human_prompt

# Streamlit app
st.title("System Prompt Generator")

# Input query
query = st.text_area("Enter your query:", height=200)

if st.button("Generate System Prompt"):
    if query.strip():
        try:
            st.write("Generating Prompt...")
            system_prompt = get_system_prompt(query)
            human_prompt = get_human_prompt(query)
            
            st.subheader("Generated System Prompt:")
            st.write(system_prompt)
            
            st.subheader("Generated Human Prompt:")
            st.write(human_prompt)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query before submitting.")


