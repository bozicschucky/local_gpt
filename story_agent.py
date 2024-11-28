from langchain_ollama import OllamaLLM
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain.agents.output_parsers import ReActSingleInputOutputParser

# Initialize Ollama
llm = OllamaLLM(model="llama3.2",  temperature=0.7, max_tokens=5000, verbose=True)


class StoryWriter:
    def __init__(self):
        self.llm = llm

    def generate_story(self, title: str, context: str) -> str:
        prompt = PromptTemplate(
            input_variables=["title", "context"],
            template="""Write a complete story based on:
            Title: {title}
            Context: {context}
            Requirements:
            - Keep it under 1000 words
            - Have a clear beginning, middle, and end
            - Include vivid descriptions
            Make the story continuous and engaging.
            keep it simple and smooth.
            """
        )

        result = self.llm.invoke(prompt.format(title=title, context=context))
        return self._format_story(title, result)

    def _format_story(self, title: str, story: str) -> str:
        return f"# {title}\n\n{story}"

    def write_story_to_file(self, title: str, story: str):
        filename = f"{title.replace(' ', '_')}.md"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(story)


def create_story_agent():
    writer = StoryWriter()

    tools = [
        Tool(
            name="WriteStory",
            func=writer.generate_story,
            description="Writes a story given a title and context"
        )
    ]

    prompt = PromptTemplate.from_template(
        """You are a creative story writer and author that uses the provided tools to write stories.
        
        Tools: {tools}
        Tool Names: {tool_names}
        
        When given a task, you will think step-by-step and then take an action.
        Thought: {input}
        Action: {action}
        {agent_scratchpad}
        """
    )

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=ReActSingleInputOutputParser()
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=True
    )



if __name__ == "__main__":
    author_agent = create_story_agent()

    title = "The Fear of the UNKNOWN, dimension"
    context = "Write a short horror story about 3 people say Jack, Matt and Ivy. Ivy is telling a story about her experience of the unknown dimension that she visited. Jack and Matt are listening to her story and they are scared. Make the descriptions vivid and engaging."

    story_result = author_agent.invoke({
        "input": f"Write a story with title '{title}' and context '{context}'",
        "action": "Write a story",
        "agent_scratchpad": "I will write a story based on the given title and context"
    })

    story_writer = StoryWriter()
    story_writer.write_story_to_file(title, story_result["output"])
    print(f"Story written to {title.replace(' ', '_')}.md")
