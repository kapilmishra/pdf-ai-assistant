import os
import re
import PyPDF2
import json
import requests
from requests.adapters import HTTPAdapter, Retry
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from tqdm.auto import tqdm

import logging
from typing import Union, Any, Optional
from pdf_ai_assistant.manager import gdrive

pdf_ext_re = re.compile(r'(.*\.pdf)')

def retry_request_session(retries: Optional[int] = 5):
    # we setup retry strategy to retry on common errors
    retries = Retry(
        total=retries,
        backoff_factor=0.1,
        status_forcelist=[
            408,  # request timeout
            500,  # internal server error
            502,  # bad gateway
            503,  # service unavailable
            504   # gateway timeout
        ]
    )
    # we setup a session with the retry strategy
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def get_file_name(query: str, handle_not_found: bool = True):
    """Get the file name from a query.

    :param query: The query to search with
    :type query: str
    :param handle_not_found: Whether to return None if no file is found,
                             defaults to True
    :type handle_not_found: bool, optional
    :return: The file name
    :rtype: str
    """
    special_chars = {
        ":": "%3A",
        "|": "%7C",
        ",": "%2C",
        " ": "+"
    }
    # create a translation table from the special_chars dictionary
    translation_table = query.maketrans(special_chars)
    # use the translate method to replace the special characters
    search_term = query.translate(translation_table)
    # init requests search session
    session = retry_request_session()
    # get the search results
    res = gdrive.GDriveAPI().listPDFFiles()
    try:
        # extract the file name
        file_name = pdf_ext_re.findall(res.text)[0]
    except IndexError:
        if handle_not_found:
            # if no file is found, return None
            return None
        else:
            # if no file is found, raise an error
            raise Exception(f'No file found for query: {query}')
    return file_name

def init_extractor(
    template: str,
    openai_api_key: Union[str, None] = None,
    max_tokens: int = 1000,
    chunk_size: int = 300,
    chunk_overlap: int = 40
):
    if openai_api_key is None and 'OPENAI_API_KEY' not in os.environ:
        raise Exception('No OpenAI API key provided')
    openai_api_key = openai_api_key or os.environ['OPENAI_API_KEY']
    # instantiate the OpenAI API wrapper
    llm = OpenAI(
        model_name='text-davinci-003',
        openai_api_key=openai_api_key,
        max_tokens=max_tokens,
        temperature=0.0
    )
    # initialize prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=['refs']
    )
    # instantiate the LLMChain extractor model
    extractor = LLMChain(
        prompt=prompt,
        llm=llm
    )
    text_splitter = tiktoken_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return extractor, text_splitter

def tiktoken_splitter(chunk_size=300, chunk_overlap=40):
    tokenizer = tiktoken.get_encoding('p50k_base')
    # create length function
    def len_fn(text):
        tokens = tokenizer.encode(
            text, disallowed_special=()
        )
        return len(tokens)
    # initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len_fn,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter


class PDF:
    refs_re = re.compile(r'\n(References|REFERENCES)\n')
    references = []
    template = """You are a master PDF reader and when given a set of references you
    always extract the most important information of the papers. For example, when
    you were given the following references:

    Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E.
    Hinton. 2016. Layer normalization. CoRR ,
    abs/1607.06450.
    Eyal Ben-David, Nadav Oved, and Roi Reichart.
    2021. PADA: A prompt-based autoregressive ap-
    proach for adaptation to unseen domains. CoRR ,
    abs/2102.12206.
    Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
    Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
    Neelakantan, Pranav Shyam, Girish Sastry, Amanda
    Askell, Sandhini Agarwal, Ariel Herbert-V oss,
    Gretchen Krueger, Tom Henighan, Rewon Child,
    Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
    Clemens Winter, Christopher Hesse, Mark Chen,
    Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin
    Chess, Jack Clark, Christopher Berner, Sam Mc-
    Candlish, Alec Radford, Ilya Sutskever, and Dario
    Amodei. 2020. Language models are few-shot learn-
    ers. In Advances in Neural Information Processing
    Systems 33: Annual Conference on Neural Informa-
    tion Processing Systems 2020, NeurIPS 2020, De-
    cember 6-12, 2020, virtual .

    You extract the following:

    Layer normalization | Lei Jimmy Ba, Jamie Ryan Kiros, Geoffrey E. Hinton | 2016
    PADA: A prompt-based autoregressive approach for adaptation to unseen domains | Eyal Ben-David, Nadav Oved, Roi Reichart
    Language models are few-shot learners | Tom B. Brown, et al. | 2020

    In the References below there are many files.

    References: {refs}

    Extracted:
    """
    llm = None

    def __init__(self, file_name: str):
        """Object to handle the extraction of an pdf file and its
        relevant information.
        
        :param file_name: The name of the pdf file to extract
        :type file_name: str
        """
        self.id = file_name
        # initialize the requests session
        self.session = requests.Session()
    
    def load(self, save: bool = True):
        """Load the pdf from the sharepoint API or from a local file
        if it already exists. Stores the pdf's text content and
        meta data in self.content and other attributes.
        
        :param save: Whether to save the pdf to a local file,
                     defaults to True
        :type save: bool, optional
        """
        # check if pdf already exists
        if os.path.exists(f'pdf/json/{self.id}.json'):
            print(f'Loading pdf/json/{self.id}.json from file')
            with open(f'pdf/json/{self.id}.json', 'r') as fp:
                attributes = json.loads(fp.read())
            for key, value in attributes.items():
                setattr(self, key, value)
        else:
            if gdrive.GDriveAPI().isPDFFileExists(self.id):
                gdrive.GDriveAPI().downloadPDFFile(self.id)
                # extract text content
                self._convert_pdf_to_text()
                # get meta for PDF
                #self._download_meta()
                if save:
                    self.save()
            else:
                return -1
            return 0

    def get_refs(self, extractor, text_splitter):
        """Get the references for the pdf.

        :param extractor: The LLMChain extractor model
        :type extractor: LLMChain
        :param text_splitter: The text splitter to use
        :type text_splitter: TokenTextSplitter
        :return: The references for the paper
        :rtype: list
        """
        if len(self.references) == 0:
            self._download_refs(extractor, text_splitter)
        return self.references
        
    def _download_refs(self, extractor, text_splitter):
        """Download the references for the pdf. Stores them in
        the self.references attribute.

        :param extractor: The LLMChain extractor model
        :type extractor: LLMChain
        :param text_splitter: The text splitter to use
        :type text_splitter: TokenTextSplitter
        """
        # get references section of pdf
        refs = self.refs_re.split(self.content)[-1]
        # we don't need the full thing, just the first page
        refs_page = text_splitter.split_text(refs)[0]
        # use LLM extractor to extract references
        out = extractor.run(refs=refs_page)
        out = out.split('\n')
        out = [o for o in out if o != '']
        # with list of references, find the pdf name
        ids = [get_file_name(o) for o in out]
        # clean up into JSONL type format
        out = [o.split(' | ') for o in out]
        # in case we're missing some fields
        out = [o for o in out if len(o) == 3]
        meta = [{
            'id': _id,
            'title': o[0],
            'authors': o[1],
            'year': o[2]
        } for o, _id in zip(out, ids) if _id is not None]
        logging.debug(f"Extracted {len(meta)} references")
        self.references = meta
    
    def _convert_pdf_to_text(self):
        """Convert the PDF to text and store it in the self.content
        attribute.
        """
        text = []
        with open(f"pdf/{self.id}", 'rb') as f:
            # create a PDF object
            pdf = PyPDF2.PdfReader(f)
            # iterate over every page in the PDF
            for page in range(len(pdf.pages)):
                # get the page object
                page_obj = pdf.pages[page]
                # extract text from the page
                text.append(page_obj.extract_text())
        text = "\n".join(text)
        self.content = text

    def save(self):
        """Save the pdf to a local JSON file.
        """
        with open(f'pdf/json/{self.id}.json', 'w') as fp:
            json.dump(self.__dict__(), fp, indent=4)

    def save_chunks(
        self,
        include_metadata: bool = False,
        path: str = "chunks"
        ):
        """Save the pdf's chunks to a local JSONL file.
        
        :param include_metadata: Whether to include the pdf's
                                 metadata in the chunks, defaults
                                 to True
        :type include_metadata: bool, optional
        :param path: The path to save the file to, defaults to "pdf"
        :type path: str, optional
        """
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}/{self.id}.jsonl', 'w') as fp:
            for chunk in self.dataset:
                if include_metadata:
                    chunk.update(self.get_meta())
                fp.write(json.dumps(chunk) + '\n')
            logging.debug(f"Saved pdf to '{path}/{self.id}.jsonl'")
    
    def get_meta(self):
        """Returns the meta information for the pdf.

        :return: The meta information for the pdf
        :rtype: dict
        """
        fields = self.__dict__()
        # drop content field because it's big
        fields.pop('content')
        return fields
    
    def chunker(self, chunk_size=300):
        # clean and split into initial smaller chunks
        clean_paper = self._clean_text(self.content)
        splitter = tiktoken_splitter(chunk_size=chunk_size)
        
        langchain_dataset = []

        paper_chunks = splitter.split_text(clean_paper)
        for i, chunk in enumerate(paper_chunks):
            langchain_dataset.append({
                'doi': self.id,
                'chunk-id': str(i),
                'chunk': chunk
            })
        logging.debug(f"Split paper into {len(paper_chunks)} chunks")
        self.dataset = langchain_dataset

    def _clean_text(self, text):
        text = re.sub(r'-\n', '', text)
        return text

    def __dict__(self):
        return {
            'id': self.id,
            'content': self.content,
             'source' : self.id
        }
    
    def __repr__(self):
        return f"PDF(file_name='{self.id}')"
