import dotenv
dotenv.load_dotenv()
ENV = dotenv.load_values()
ENV = dotenv.dotenv_values()
import openaiENV['OPENAI_SECRET_KEY']
OPENAI_SECRET_KEY = ENV['OPENAI_SECRET_KEY']
import openai
openai.api_key = OPENAI_SECRET_KEY
openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant that never lies and always admits when it isn't sure about something."},
        {"role": "user", "content": "Give me an example of a deeply ekphrastic poem about the Getty Museum architecture."},
    ]
)
openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant that never lies and always admits when it isn't sure about something."},
        {"role": "user", "content": "Give me an example of a deeply ekphrastic poem about the Getty Museum architecture."},
    ]
)
!nano .env
ENV = dotenv.dotenv_values()
ENV
OPENAI_SECRET_KEY = ENV['OPENAI_SECRET_KEY']
openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant that never lies and always admits when it isn't sure about something."},
        {"role": "user", "content": "Give me an example of a deeply ekphrastic poem about the Getty Museum architecture."},
    ]
)
