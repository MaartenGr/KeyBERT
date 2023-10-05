A minimal method for keyword extraction with Large Language Models (LLM). There are a number of implementations that allow you to mix and match `KeyBERT` with `KeyLLM`. You could also choose to use `KeyLLM` without `KeyBERT`.

<div class="excalidraw">
--8<-- "docs/images/keyllm.svg"
</div>

We start with an example of some data:

```python
documents = [
"The website mentions that it only takes a couple of days to deliver but I still have not received mine.",
"I received my package!",
"Whereas the most powerful LLMs have generally been accessible only through limited APIs (if at all), Meta released LLaMA's model weights to the research community under a noncommercial license."
]
```

This data was chosen to show the different use cases and techniques. As you might have noticed documents 1 and 2 are quite similar whereas document 3 is about an entirely different subject. This similarity will be taken into account when using `KeyBERT` together with `KeyLLM`

Let's start with `KeyLLM` only. 


# Use Cases

If you want the full performance and easiest method, you can skip the use cases below and go straight to number 5 where you will combine `KeyBERT` with `KeyLLM`.

!!! Tip
    If you want to use KeyLLM without any of the HuggingFace packages, you can install it as follows:
    `pip install keybert --no-deps`
    `pip install scikit-learn numpy rich tqdm`
    This will make the installation much smaller and the import much quicker.

## 1. **Create** Keywords with `KeyLLM`

We start by creating keywords for each document. This creation process is simply asking the LLM to come up with a bunch of keywords for each document. The focus here is on **creating** keywords which refers to the idea that the keywords do not necessarily need to appear in the input documents.

Install the relevant LLM first:

```bash
pip install openai
```

Then we can use any OpenAI model, such as ChatGPT, as follows:

```python
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM

# Create your LLM
openai.api_key = "sk-..."
llm = OpenAI()

# Load it in KeyLLM
kw_model = KeyLLM(llm)

# Extract keywords
keywords = kw_model.extract_keywords(documents)
```

This creates the following keywords:

```python
[['Website',
  'Delivery',
  'Mention',
  'Timeframe',
  'Not received',
  'Order fulfillment'],
 ['Package', 'Received', 'Delivery', 'Order fulfillment'],
 ['Powerful LLMs',
  'Limited APIs',
  'Meta',
  'Model weights',
  'Research community',
  '']]
```

## 2. **Extract** Keywords with `KeyLLM`

Instead of creating keywords out of thin air, we ask the LLM to check whether they actually appear in the text and limit the keywords to those that are found in the documents. We do this by using a custom prompt together with `check_vocab=True`:

```python
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM

# Create your LLM
openai.api_key = "sk-..."

prompt = """
I have the following document:
[DOCUMENT]

Based on the information above, extract the keywords that best describe the topic of the text.
Make sure to only extract keywords that appear in the text.
Use the following format separated by commas:
<keywords>
"""
llm = OpenAI()

# Load it in KeyLLM
kw_model = KeyLLM(llm)

# Extract keywords
keywords = kw_model.extract_keywords(documents, check_vocab=True); keywords
```

This creates the following keywords:

```python
[['website', 'couple of days', 'deliver', 'received'],
 ['package', 'received'],
 ['LLMs',
  'APIs',
  'Meta',
  'LLaMA',
  'model weights',
  'research community',
  'noncommercial license']]
```

## 3. **Fine-tune** Candidate Keywords

If you already have a list of keywords, you could fine-tune them by asking the LLM to come up with nicer tags or names that we could use.  We can use the `[CANDIDATES]` tag in the prompt to assign where they should go.

```python
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM

# Create your LLM
openai.api_key = "sk-..."

prompt = """
I have the following document:
[DOCUMENT]

With the following candidate keywords:
[CANDIDATES]

Based on the information above, improve the candidate keywords to best describe the topic of the document.

Use the following format separated by commas:
<keywords>
"""
llm = OpenAI(model="gpt-3.5-turbo", prompt=prompt, chat=True)

# Load it in KeyLLM
kw_model = KeyLLM(llm)

# Extract keywords
candidate_keywords = [['website', 'couple of days', 'deliver', 'received'],
 ['received', 'package'],
 ['most powerful LLMs',
  'limited APIs',
  'Meta',
  "LLaMA's model weights",
  'research community',
  'noncommercial license']]
keywords = kw_model.extract_keywords(documents, candidate_keywords=candidate_keywords); keywords
```

This creates the following keywords:

```python
[['delivery timeframe', 'discrepancy', 'website', 'order status'],
 ['received package'],
 ['most powerful language models',
  'API limitations',
  "Meta's release",
  "LLaMA's model weights",
  'research community access',
  'noncommercial licensing']]
```

## 4. **Efficient** `KeyLLM`

If you have embeddings of your documents, you could use those to find documents that are most similar to one another. Those documents could then all receive the same keywords and only one of these documents will need to be passed to the LLM. This can make computation much faster as only a subset of documents will need to receive keywords.

<div class="excalidraw">
--8<-- "docs/images/efficient.svg"
</div>

!!! Tip
    Before you get started, it might be worthwhile to uninstall sentence-transformers and re-install it from the main branch. 
    There is an issue with community detection (cluster) that might make the model run without finishing. It is as straightforward as:
    `pip uninstall sentence-transformers`
    `pip install --upgrade git+https://github.com/UKPLab/sentence-transformers`


```python
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM
from sentence_transformers import SentenceTransformer

# Extract embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents, convert_to_tensor=True)

# Create your LLM
openai.api_key = "sk-..."
llm = OpenAI()

# Load it in KeyLLM
kw_model = KeyLLM(llm)

# Extract keywords
keywords = kw_model.extract_keywords(documents, embeddings=embeddings, threshold=.75)
```

This creates the following keywords:

```python
[['Website',
  'Delivery',
  'Mention',
  'Timeframe',
  'Not received',
  'Waiting',
  'Order fulfillment'],
 ['Received', 'Package', 'Delivery', 'Order fulfillment'],
 ['Powerful LLMs', 'Limited APIs', 'Meta', 'LLaMA', 'Model weights']]
```


## 5. **Efficient** `KeyLLM` + `KeyBERT`

This is the best of both worlds. We use `KeyBERT` to generate a first pass of keywords and embeddings and give those to `KeyLLM` for a final pass. Again, the most similar documents will be clustered and they will all receive the same keywords. You can change this behavior with `threshold`. A higher value will reduce the number of documents that are clustered and a lower value will increase the number of documents that are clustered.

<div class="excalidraw">
--8<-- "docs/images/keybert_keyllm.svg"
</div>

!!! Tip
    Before you get started, it might be worthwhile to uninstall sentence-transformers and re-install it from the main branch. 
    There is an issue with community detection (cluster) that might make the model run without finishing. It is as straightforward as:
    `pip uninstall sentence-transformers`
    `pip install --upgrade git+https://github.com/UKPLab/sentence-transformers`


```python
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM, KeyBERT

# Create your LLM
openai.api_key = "sk-..."
llm = OpenAI()

# Load it in KeyLLM
kw_model = KeyBERT(llm=llm)

# Extract keywords
keywords = kw_model.extract_keywords(documents); keywords
```

This creates the following keywords:

```python
[['Website',
  'Delivery',
  'Timeframe',
  'Mention',
  'Order fulfillment',
  'Not received',
  'Waiting'],
 ['Package', 'Received', 'Confirmation', 'Delivery', 'Order fulfillment'],
 ['LLMs', 'Limited APIs', 'Meta', 'LLaMA', 'Model weights', '']]
```
