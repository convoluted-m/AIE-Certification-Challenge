from langchain_openai import ChatOpenAI

# Make sure OPENAI_API_KEY is set in your environment before running this.
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

resp = llm.invoke("Say 'OpenAI is working!' in exactly 3 words.")
print(resp.content)