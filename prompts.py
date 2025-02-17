from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
    PromptTemplate
)

# Define a prompt template
prompt_template = """
### System Instruction: You are a Professional Automobile Engineer, Read and understand the following context to answer the upcoming question. Respond in 15 to 20 output tokens.  

---

### Retrieved Context: {context}

---

### User Query: {question}

---

### Response Guidelines:
1. **Prioritize Retrieved Context** - Use the provided context as the primary source of truth. Summarize and synthesize the most relevant information.
2. **Bridge Knowledge Gaps** - If the retrieved context is insufficient, supplement it with your own knowledge, but clearly indicate when you are doing so.
3. **Ensure Accuracy & Clarity** - Provide a clear, structured, and fact-based answer. Avoid speculation.
4. **Cite Sources** - If the retrieved documents contain useful references, include them in your response.
5. **Structured Output** - Format responses in a well-organized manner (e.g., bullet points, step-by-step instructions, or concise summaries).
6. **User-Friendly Language** - Simplify complex concepts when needed, ensuring accessibility for all users.

---

### Final Response:
(Generate preise and short response based on the above instructions.)

"""

# ### Output Format:
# ```json
# {{
#   "query": "{question}",
#   "answer": "<Provide a well-structured response here>",
#   "sources": ["<List sources from retrieved context, if available>"],
#   "additional_notes": "<If retrieved context is insufficient, provide insights based on general knowledge and indicate it>"
# }}

# ---

# prompt = prompt_template.format(context=context, question=input_query)
rag_prompt_custom = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
    )



_template = """
[INST] 
Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language, 
that can be used to query a FAISS index. This query will be used to retrieve documents with additional context.

Let me share a couple examples.

If you do not see any chat history, you MUST return the "Follow Up Input" as is:
```
Chat History:
Follow Up Input: How is Lawrence doing?
Standalone Question:
How is Lawrence doing?
```

If this is the second question onwards, you should properly rephrase the question like this:
```
Chat History:
Human: How is Lawrence doing?
AI: 
Lawrence is injured and out for the season.
Follow Up Input: What was his injury?
Standalone Question:
What was Lawrence's injury?
```

Now, with those examples, here is the actual chat history and input question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
[your response here]
[/INST] 
"""

STANDALONE_QUESTION_PROMPT = PromptTemplate.from_template(_template)



contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

