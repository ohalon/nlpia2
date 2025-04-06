from langchain import HuggingFaceHub

HUGGING_FACE_API_KEY =

llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={
        "temperature": 0.3,
        "max_length": 200},
    huggingfacehub_api_token='<your_API_token>')

from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain

prompt = PromptTemplate(
   input_variables=["message"],
   template="{message}")

chain = LLMChain(llm=llm, prompt=prompt)

chain.predict(message="Hi Bot!")

template = """
    This is a conversation between a human and a
    chatbot. The chatbot is friendly and provides
    answers based on the previous conversation and
    the context.

    Human says: {message}
    Chatbot responds:
    """

prompt = PromptTemplate(
    input_variables=["message"],  # <1>
    template=template)

chain = LLMChain(
    llm=llm, verbose=True, prompt=prompt  # <2>
    )

chain.predict(message="Hi Bot! My name is Maria.")

chain.predict(message="What is my name?")

template = """
    This is a conversation between a human and a chatbot.
    The chatbot is friendly and provides answers based
    on the previous conversation and the context."

    {chat_history}
    Human says: {message}
    Chatbot responds:"""

memory = ConversationBufferMemory(
    memory_key='chat_history')  # <1>

chain = LLMChain(llm=llm, memory=memory, prompt=prompt)

convo_chain = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory
    )

convo_chain.prompt.template

chain.predict(message="Hi chatbot! My name is Maria")

chain.predict(message="What is my name?")

message = """
    I have a brother Sergey.
    He and his wife Olga live in Tel Aviv.
    What's the name of my sister-in-law?"""

chain.predict(message=message)

from langchain.llms import Replicate

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN = os.environ["REPLICATE_API_TOKEN"]

llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:" +
    "df7690",  # <1>
    input={
        "temperature": 0.5,
        "max_length": 350,
        "top_p": 1,
    })

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""This is a conversation between a math teacher and
    a third-grade student. The teacher asks math questions of the
    student and evaluates the student's answer one at a time.

    Complete the conversation with only one response at a time.

    {history}
    student:{input}
    teacher:""")

memory = ConversationBufferMemory(
    memory_key='history',
    ai_prefix='teacher',  # <1>
    human_prefix='student',  # <2>
    )

memory.save_context(
    {"input": "Ask me math questions!"},
    {"output": "Sure, let's do it! 9,10,11?"})

memory.save_context(
    {"input": "12"},
    {"output": "Perfect! 38,39,40?"})

memory.save_context(
    {"input": "42"},
    {"output": "Oops. Not quite. Try again."})

memory.save_context(
    {"input": "41"},
    {"output": "Good work! 2,4,6?"})

math_convo = ConversationChain(llm=llm, memory=memory)

math_convo.prompt = prompt

math_convo.predict(input="9")

math_convo = ConversationChain(
    llm=llm, memory=memory)  # <1>

math_convo.predict(input="9")

prompt = PromptTemplate(
   input_variables=["history", "input"],
   template="""You are a math teacher that's teaching math
   to a third-grade student.Prompt the student to complete number
   sequences from the following list and compare their answer
   with the last number in the following sequences:
     - 9,10,11,12
     - 38,39,40,41
     - 2,4,6,8
     - 1,5,9,13

   {history}
   student:{input}
   teacher:"""

math_chatbot=MathChatbot(prompt)

math_chatbot.answer("9")

math_chatbot=MathChatbot(prompt)

math_chatbot.answer("9")
