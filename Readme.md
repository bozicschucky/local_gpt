# Local GPT

This project is a chat application that uses any GPT model provided by ollama. It allows users to upload PDF and image files and add those files to the chat context. The application is built using Streamlit and integrates with a local GPT model using Ollama as the GPT model server.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Packages to install:
```
Python 3.8+
 Streamlit
 SQLAlchemy
 PyPDF2
 Pillow
 dotenv
 langChain
```



### Installing

A step-by-step series of examples that tell you how to get a development environment running:

1. Clone the repository:
```
git clone https://github.com/bozicschucky/local-gpt-chat.git cd local-gpt-chat
```


2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Set up environment variables:

Create a `.env` file in the project root and add the following:
```
llm_model=your_model_name
```

5. Run the application:
```
streamlit run local_gpt.py
```


## Deployment

Add additional notes about how to deploy this on a live system.

## Built With

* [Streamlit](https://streamlit.io/) - The web framework used
* [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM
* [PyPDF2](https://pypi.org/project/PyPDF2/) - PDF processing
* [Pillow](https://python-pillow.org/) - Image processing
* [dotenv](https://pypi.org/project/python-dotenv/) - Environment variable management

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Charles  ssekitto** - *Initial work* - [github_profile](https://github.com/bozicschucky)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Hat tip to langchain and ollama docs
* Inspiration
* etc.