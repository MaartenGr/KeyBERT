# Large Language Models (LLM)
In this tutorial we will be going through the Large Language Models (LLM) that can be used in KeyLLM.
Having the option to choose the LLM allow you to leverage the model that suit your use-case.

### **OpenAI**
To use OpenAI's external API, we need to define our key and use the `keybert.llm.OpenAI` model.

We install the package first:

```bash
pip install openai
```

Then we run OpenAI as follows:

```python
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM

# Create your OpenAI LLM
client = openai.OpenAI(api_key=MY_API_KEY)
llm = OpenAI(client)

# Load it in KeyLLM
kw_model = KeyLLM(llm)

# Extract keywords
keywords = kw_model.extract_keywords(MY_DOCUMENTS)
```

If you want to use a chat-based model, please run the following instead:

```python
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM

# Create your LLM
client = openai.OpenAI(api_key=MY_API_KEY)
llm = OpenAI(client, model="gpt-3.5-turbo", chat=True)

# Load it in KeyLLM
kw_model = KeyLLM(llm)
```

### **Cohere**
To use Cohere's external API, we need to define our key and use the `keybert.llm.Cohere` model.

We install the package first:

```bash
pip install cohere
```

Then we run Cohere as follows:


```python
import cohere
from keybert.llm import Cohere
from keybert import KeyLLM

# Create your OpenAI LLM
co = cohere.Client(my_api_key)
llm = Cohere(co)

# Load it in KeyLLM
kw_model = KeyLLM(llm)

# Extract keywords
keywords = kw_model.extract_keywords(MY_DOCUMENTS)
```

### **LiteLLM**
[LiteLLM](https://github.com/BerriAI/litellm) allows you to use any closed-source LLM with KeyLLM

We install the package first:

```bash
pip install litellm
```


Let's use OpenAI as an example:

```python
import os
from keybert.llm import LiteLLM
from keybert import KeyLLM

# Select LLM
os.environ["OPENAI_API_KEY"] = "sk-..."
llm = LiteLLM("gpt-3.5-turbo")

# Load it in KeyLLM
kw_model = KeyLLM(llm)
```

### 🤗 **Hugging Face Transformers**
To use a Hugging Face transformers model, load in a pipeline and point
to any model found on their model hub (https://huggingface.co/models). Let's use Llama 2 as an example:

```python
from torch import cuda, bfloat16
import transformers

model_id = 'meta-llama/Llama-2-7b-chat-hf'

# 4-bit Quantization to load Llama 2 with less GPU memory
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# Llama 2 Model & Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
)
model.eval()

# Our text generator
generator = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,
    max_new_tokens=500,
    repetition_penalty=1.1
)
```

Then, we load the `generator` in `KeyLLM` with a custom prompt:

```python
from keybert.llm import TextGeneration
from keybert import KeyLLM

prompt = """
<s>[INST] <<SYS>>

You are a helpful assistant specialized in extracting comma-separated keywords.
You are to the point and only give the answer in isolation without any chat-based fluff.

<</SYS>>
I have the following document:
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST] meat, beef, eat, eating, emissions, steak, food, health, processed, chicken [INST]

I have the following document:
- [DOCUMENT]

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST]
"""

# Load it in KeyLLM
llm = TextGeneration(generator, prompt=prompt)
kw_model = KeyLLM(llm)
```

### **LangChain**

To use `langchain` LLM client in KeyLLM, we can simply load in any LLM in `langchain` and pass that to KeyLLM.

We install langchain and corresponding LLM provider package first. Take OpenAI as an example:

```bash
pip install langchain
pip install langchain-openai # LLM provider package
```
> [!NOTE]
> KeyBERT only supports `langchain >= 0.1`


Then create your LLM client with `langchain`


```python
from langchain_openai import ChatOpenAI

_llm = ChatOpenAI(
    model="gpt-4o",
    api_key="my-openai-api-key",
    temperature=0,
)
```

Finally, pass the `langchain` llm client to KeyBERT as follows:

```python
from keybert.llm import LangChain
from keybert import KeyLLM

# Create your LLM
llm = LangChain(_llm)

# Load it in KeyLLM
kw_model = KeyLLM(llm)

# Extract keywords
keywords = kw_model.extract_keywords(MY_DOCUMENTS)
```
