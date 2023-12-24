CLIP Model in a Nutshell
CLIP is a fancy model that's been trained to understand images and text together. It does this via embeddings.

Embeddings are just a vector of floats. In clip's case, there are 512 dimensions.

# How to generate embeddings using subnet 19

## Text Embedding
To embed text, you can use the ClipEmbeddingTexts synapse. It lives in here: [Protocol.py file](https://github.com/namoray/vision/blob/main/template/protocol.py)

You can do something similar to the following:
```python
import bittensor as bt
from template.protocol import ClipEmbeddingTexts
import random

dendrite = bt.dendrite()
metagraph = bt.metagraph(19)

text = "Give me an image of a dog in sunglasses"
synapse = ClipEmbeddingTexts(text_prompts=[text])
axon = random.choice(metagraph.axons)
text_embedding = await dendrite.forward(axon, synapse)
```
Make sure you're in an environment that can run async code, for example, try running the above in a jupyter notebook.

## Image Embedding
To embed images, you can use the ClipEmbeddingImages synapse. It lives in here: [Protocol.py file](https://github.com/namoray/vision/blob/main/template/protocol.py)

You can do something similar to the following:
```python
import bittensor as bt
from template.protocol import ClipEmbeddingImages
from core import utils
import random

dendrite = bt.dendrite()
metagraph = bt.metagraph(19)

image = await utils.get_random_image(1200, 1200)
synapse = ClipEmbeddingImages(image_b64s=[image])
axon = random.choice(metagraph.axons)
image_embedding = await dendrite.forward(axon, synapse)
```
Make sure you're in an environment that can run async code, for example, try running the above in a jupyter notebook.

# How do I use a bunch of embeddings to perform an 'image search'
All you need to is compare all the embeddings and find the closest match.

For example if you have something like this:
```python
import numpy as np

a_bunch_of_image_embeddings = [...]  # 2D list of embeddings, each list is an embedding of dimension 512
text_embedding_to_search_with = [...]  # 1D list of text embedding of dimension 512

text_embedding_array = np.array(text_embedding_to_search_with)
image_embeddings_array = np.array(image_embeddings)

# I would highly highly advise normalising - it gives much better results
text_embedding_normalized = text_embedding_array / np.linalg.norm(text_embedding_array) 
image_embeddings_normalized = image_embeddings_array / np.linalg.norm(image_embeddings_array, axis=1, keepdims=True)

dot_product_scores = np.dot(image_embeddings_normalized, text_embedding_normalized)
index_of_closest_match = np.argmax(dot_product_scores)
```

# Store them in a vector database or similar (advised)
I would advise you instead store all the image embeddings in something like FAISS, and instead use that to do something similar to the above.
You could even use a hosted embeddings service if you like, like [pinecone + FAISS](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/) 


# Any examples of the usecases of this?
Of course - here's are example apps built on subnet 19 segmentation:

- Corcel image search: app.corcel.io/image-search
- Your app goes here ( please message me if you want any support building / when you have finished building, and I will add it here! )