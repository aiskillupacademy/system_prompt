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
def get_system_prompt(query):
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
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain.run(query)

# Streamlit app
st.title("System Prompt Generator")

# Input query
query = st.text_area("Enter your query:", height=200)

if st.button("Generate System Prompt"):
    if query.strip():
        try:
            st.write("Generating system prompt...")
            system_prompt = get_system_prompt(query)
            st.subheader("Generated System Prompt:")
            st.code(system_prompt, language="text")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query before submitting.")


