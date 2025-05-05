# Resume_Screening_System
ABSTRACT

In today’s fast-paced, technology-driven hiring landscape, Human Resource departments face the challenge of managing and evaluating a massive volume of resumes for each job opening. Manual screening processes are not only time-consuming but also prone to human error and unconscious bias, which can result in the dismissal of well-qualified candidates. As organizations seek to streamline recruitment and reduce time-to-hire, Artificial Intelligence (AI) and Natural Language Processing (NLP) have emerged as transformative tools. This project proposes an AI-powered Resume Screening System that utilizes the all-MiniLM-L6-v2 model from the Sentence Transformers library to semantically evaluate and rank resumes based on their relevance to a given job description. By extracting text from PDF and DOCX files, cleaning and preprocessing the content, and embedding both job descriptions and resumes into vector representations, the system effectively compares them using cosine similarity.

The core objective of the system is to automate resume filtering and present recruiters with a ranked list of candidates based on contextual relevance, allowing them to focus on higher-value tasks such as interviews and candidate engagement. Unlike traditional keyword-based filters, this approach captures deeper semantic relationships in text, enabling recognition of relevant skills and experience regardless of wording differences. The system is lightweight, scalable, and capable of delivering real-time results even on modest computational resources, making it suitable for enterprise-level and startup use alike. Ultimately, this solution demonstrates the practical and ethical potential of applying modern NLP to recruitment, paving the way for more intelligent, inclusive, and efficient hiring processes.

INTRODUCTION
The recruitment process has significantly evolved over the last decade due to digital technologies. However, a major challenge persists: the initial screening of resumes. Hiring managers often manually review hundreds or thousands of resumes, which is a time-consuming and biased process that can lead to the overlooking of qualified candidates.
To address this issue, we introduce a Resume Screening System that employs advanced Natural Language Processing (NLP) techniques. This system uses the all-MiniLM-L6-v2 model, a transformer-based language representation model from the Sentence Transformers. This model converts text into fixed-size embeddings, effectively capturing semantic meaning and enhancing applications like semantic similarity, ranking, and information retrieval. The system extracts raw text from resumes in PDF or DOCX format using libraries like pdfminer and docx2txt. The text is processed to remove unnecessary characters, standardize formatting, and convert it to lowercase. It is then embedded into a high-dimensional vector space using the all-MiniLM-L6-v2 model. 
The job description provided by the recruiter undergoes the same embedding process. The cosine similarity between the job description and each resume is calculated to determine relevance scores, allowing resumes to be ranked in descending order of match quality. This enables recruiters to quickly identify the most suitable candidates by going beyond simple keyword matching and understanding the context of the content, thereby improving the quality of matches. 
This approach offers several advantages over traditional rule-based or keyword-matching systems. It enhances recall by capturing semantically similar terms, reducing the chances of overlooking resumes due to phrasing variations. Additionally, it improves precision by minimizing false positives-resumes that contain relevant keywords but lack context. The system is adaptable and scalable, suitable for both small startups and large enterprises. Its lightweight all-MiniLM-L6-v2 model allows for near real-time results without heavy GPU infrastructure, making it practical for various recruitment platforms.
In summary, the Resume Screening System showcases how modern natural language processing (NLP) models can automate and enhance the hiring process, providing an efficient solution to the challenges of manual resume screening.


![Image](https://github.com/user-attachments/assets/0c572859-1d2d-4baf-be9b-9ad7ed09ebde)

![Image](https://github.com/user-attachments/assets/e75b551f-7dbb-4894-828a-92d711090847)


![Image](https://github.com/user-attachments/assets/442d6ee3-a432-4cdb-805d-696dc7678f97)


The objective of this project is to create an AI-driven resume screening system that automatically evaluates and ranks candidate resumes based on their relevance to specific job descriptions. The system utilizes the all-MiniLM-L6-v2 model from the Sentence-Transformers library to compute semantic similarities between resumes and job descriptions, enabling efficient and intelligent candidate filtering.
1. Preprocessing and Embedding Generation
The first phase of the methodology involves extracting and preprocessing text from resumes and job descriptions:
a)	Text Extraction: Resumes in .pdf and .docx formats are parsed using pdfminer and docx2txt, respectively. A utility function scans the specified folder and extracts raw textual content from each file.
b)	Cleaning: The extracted text undergoes preprocessing, such as lowercasing and whitespace normalization, to ensure uniformity and reduce noise.
c)  Embedding Model: The cleaned resume text and job description are encoded using the all-MiniLM-L6-v2 model. This transformer-based model generates high-quality sentence embeddings that capture semantic meaning in a compact 384-dimensional vector.
This stage ensures that both resumes and job descriptions are represented in a common semantic vector space, suitable for similarity computation.

2. Similarity Computation and Ranking:
Once the textual data has been embedded, the second phase performs semantic matching between resumes and the job description:
a)	Cosine Similarity: For each resume embedding, cosine similarity is computed against the job description embedding. This metric quantifies the semantic closeness between two text vectors, with a score closer to 1 indicating higher similarity.
![Image](https://github.com/user-attachments/assets/f166d37d-7499-4fb9-8eec-9af8aed0a837)

b)	Ranking: The system then sorts all resumes based on their similarity scores in descending order. This ranked list provides a prioritized view of candidates most relevant to the job, effectively automating the screening process.


IMPLEMENTATION DETAILS


ALGORITHM:
PDFMiner Text Extraction Algorithm (file):
Step 1: Open and Read the PDF File
	1.1: Load the PDF file using PDF parser that reads the binary data
	1.2: Create a PDF document object (PDFDocument) from the parser
	1.3: Check if document allows text extraction using is_extractable
Step 2: Interpret Document Structure
2.1: A PDF resource manager (PDFResourceManager) to manage shared resources like fonts and images.
2.2: A layout analyser (LAParams) to analyse layout and spacing between characters, words, and lines.
Step 3: Process Each Page
	3.1: For each page:
3.1.1: Create a PDFPageInterpreter, passing in the resource manager and layout analyser.
3.1.2: Feed the page into the interpreter, which builds a layout tree representing the visual structure.
			[End of for each loop]
	Step 4: Extract Text
		4.1: Traverse this tree and collect text from text container elements.
	Step 5: Accumulate the extracted text into a string.
	Step 6: Stop



All-MiniLM-L6-v2 Embedding Model (Sentence):
Step 1: Input the sequence of sentences
Step 2: For each sentence:
	2.1: Apply BERT Tokeniser (to convert it into a sequence of subword tokens)
[End of For Each Loop]
Step 3: For each token
	3.1: Generate Initial Embedding Vectors
Step 4: Pass the sequence of token embedding through 6 layers of the Transformer encoder
	4.1: Each layer contains:
		4.1.1: Calculate Attention Score for each token relative to all other tokens.
	4.2: Apply a Feedforward network to the outputs of the self-attention mechanism
[Step 4 is repeated 6 times]
Step 5: Obtain the final token embeddings for each token 
Step 6: Average the final token embeddings to obtain a single embedding for the sentence.
Step 7: Return the sentence embedding for each sentence.
Step 8: Stop

![Image](https://github.com/user-attachments/assets/041e008e-d4be-47f6-a392-dc83c2703f98)

PSEUDO-CODE OF PROPOSED SYSTEM
1. LOAD pre-trained transformer model for sentence embeddings
2. FUNCTION clean_text(text):
    2.1: CONVERT text to lowercase
    2.2: REPLACE multiple whitespaces with a single space
    2.3: TRIM leading and trailing spaces
    2.4: RETURN cleaned text
3. FUNCTION extract_text_from_file(file_path):
   3.1: IF the file extension is .pdf:
        3.1.1: EXTRACT and RETURN text from PDF
    ELSE IF file extension is .docx:
        3.1.2: EXTRACT and RETURN text from DOCX
    ELSE:
        3.1.3: RETURN empty string
4. FUNCTION load_resumes(folder_path):
    4.1: INITIALISE empty dictionary resumes
    4.2: FOR each file in folder_path:
        4.2.1: IF file is .pdf or .docx:
            4.2.1.1: full_path ← combine folder_path and file name
            4.2.1.2: text ← extract_text_from_file(full_path)
            4.2.1.3: cleaned_text ← clean_text(text)
            4.2.1.4: ADD cleaned_text to resumes with filename as key
    4.3: RETURN resumes dictionary

5.FUNCTION load_job_description(jd_path):
    5.1: OPEN job description file
    5.2: READ content into text
    5.3: CLEAN text using the clean_text function
    5.4: RETURN cleaned job description text
6. FUNCTION compute_similarity(resumes_dict, jd_text):
    6.1: ENCODE job description text into embedding vector
    6.2: INITIALISE empty result dictionary
    6.3: FOR each resume in resumes_dict:
        6.3.1: ENCODE resume text into embedding vector
        6.3.2: CALCULATE cosine similarity between job description and resume embeddings
        6.3.3: CONVERT similarity to percentage and ROUND to 2 decimal places
        6.3.4: STORE percentage similarity in result dictionary with filename as key
    6.4: SORT the result dictionary in descending order of similarity
    6.5: RETURN sorted result dictionary

RESULT AND DISCUSSION
    ![Image](https://github.com/user-attachments/assets/c403f97e-1713-42c0-936c-8c0a3bb774d2)
    ![Image](https://github.com/user-attachments/assets/54f001d9-48f8-4198-8810-68f80760d459)

CONCLUSION AND FUTURE SCOPE

The Resume Screening System, developed using the all-MiniLM-L6-v2 model presents a robust and efficient solution to the challenges of modern recruitment. By transforming both job descriptions and resumes into semantically rich vector representations, the system enables meaningful comparisons that go beyond keyword matching. This results in more accurate and context-aware resume shortlisting.
The use of a lightweight yet powerful transformer model ensures fast performance without compromising on accuracy, making it suitable for both small-scale and enterprise-level hiring processes. Additionally, the modular design of the system allows for easy integration, extension, and customization based on specific organizational needs.
Overall, this system streamlines the recruitment workflow, reduces manual overhead, and enhances the objectivity and fairness of candidate evaluation. 
With further improvements such as multi-lingual support, domain-specific fine-tuning, it holds the potential to become a cornerstone tool in intelligent talent acquisition.


REFERNCES


[1] Prathima, V., Singh, A., Rohilla, T., Rebbavarapu, V., & Anjum, N. (2024). Resume Application Tracking System with Google Gemini Pro. International Journal for Research in Applied Science and Engineering Technology.
[2] Anand, S., & Giri, A. K. (2024). Optimizing Resume Design for ATS Compatibility: A Large Language Model Approach. International Journal on Smart & Sustainable Intelligent Computing, 1(2), 49-57.
[3] Abdelhay, S., AlTalay, M. S. R., Selim, N., Altamimi, A. A., Hassan, D., Elbannany, M., & Marie, A. (2025). The impact of generative AI (Chatgpt) on recruitment efficiency and candidate quality: The mediating role of process automation level and the moderating role of organizational size. Frontiers in Human Dynamics, 6, 1487671.
[4] Cohen, L., Hsieh, J., Hong, C., & Shen, J. H. (2025). Two Tickets are Better than One: Fair and Accurate Hiring Under Strategic LLM Manipulations. arXiv preprint arXiv:2502.13221
[5] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT networks. arXiv preprint arXiv:1908.10084.
[6] Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. Advances in neural information processing systems, 33, 5776-5788.
[7] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019, June). Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers) (pp. 4171-4186).
[8] Kusner, M., Sun, Y., Kolkin, N., & Weinberger, K. (2015, June). From word embeddings to document distances. In International conference on machine learning (pp. 957-966). PMLR.
