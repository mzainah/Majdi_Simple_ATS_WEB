from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Common words to exclude
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
    'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once',
    # Job-specific common words
    'experience', 'work', 'year', 'years', 'working', 'job', 'role', 'team',
    'company', 'position', 'skills', 'ability', 'candidate', 'required',
    'requirements', 'qualified', 'qualification', 'proficient', 'proficiency',
    'responsible', 'responsibilities', 'etc', 'including', 'include', 'includes',
    'various', 'using', 'use', 'used', 'must', 'need', 'needs', 'needed',
    'prefer', 'preferred', 'strong', 'excellent'
}

def tokenize_text(text):
    """Simple tokenization using regex"""
    # Convert to lowercase and split on non-word characters
    words = re.findall(r'\b\w+\b', text.lower())
    # Remove stopwords and short words
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def extract_technical_skills(text):
    """Extract technical skills using regex patterns"""
    # tech_patterns = {
    #     # Programming Languages
    #     r'\b(python|java|javascript|typescript|ruby|php|cpp|c\+\+|c#|go|rust|swift|kotlin)\b',
    #     # Web Technologies
    #     r'\b(html5?|css3?|react|angular|vue|node|express|django|flask|spring|laravel|rails)\b',
    #     # Databases
    #     r'\b(sql|mysql|postgresql|mongodb|redis|oracle|sqlite)\b',
    #     # Cloud & DevOps
    #     r'\b(aws|azure|gcp|docker|kubernetes|jenkins|git|ci/cd)\b',
    #     # Data Science
    #     r'\b(tensorflow|pytorch|pandas|numpy|scikit-learn|ml|ai|nlp)\b'
    # }
    
    tech_patterns = {
        # Programming Languages
        r'\b(python|java|javascript|typescript|ruby|php|cpp|c\+\+|c#|go|rust|swift|kotlin|scala|perl|r|bash|shell|powershell)\b',
        
        # Web Technologies
        r'\b(html5?|css3?|react|angular|vue|node\.?js|express|django|flask|spring|laravel|rails|sass|less|bootstrap|webpack|graphql|restapi|soap)\b',
        
        # Databases
        r'\b(sql|mysql|postgresql|mongodb|redis|oracle|sqlite|mariadb|cassandra|dynamodb|firebase|cosmosdb|neo4j|hbase)\b',
        
        # Cloud & DevOps
        r'\b(aws|azure|gcp|docker|kubernetes|jenkins|git|ci/cd|terraform|ansible|puppet|chef|vagrant|prometheus|grafana|istio|helm|nginx|apache)\b',
        
        # Data Science & Machine Learning
        r'\b(tensorflow|pytorch|pandas|numpy|scikit-learn|ml|ai|nlp|opencv|keras|spark|hadoop|hive|pig|tableau|powerbi|matplotlib|seaborn|plotly)\b',
        
        # Mobile Development
        r'\b(android|ios|flutter|react native|xamarin|swiftui|kotlin multiplatform)\b',
        
        # Software Development Tools
        r'\b(jira|confluence|trello|slack|bitbucket|github|gitlab|vscode|intellij|eclipse|sublime|atom|vim|emacs)\b',
        
        # Networking & Security
        r'\b(tcp/ip|udp|dns|http|https|ssl|tls|vpn|firewall|ids|ips|owasp|penetration testing|encryption|authentication|authorization)\b',
        
        # Operating Systems
        r'\b(linux|unix|windows|macos|ubuntu|centos|debian|fedora|redhat)\b',
        
        # Version Control & Collaboration
        r'\b(git|svn|mercurial|bitbucket|github|gitlab|pull requests|code review|agile|scrum|kanban)\b',
        
        # Testing & QA
        r'\b(selenium|junit|testng|pytest|mocha|chai|jest|cypress|postman|soapui|load testing|performance testing|qa|automation testing)\b',
        
        # Big Data & Analytics
        r'\b(big data|hadoop|spark|hive|pig|kafka|storm|flink|elasticsearch|logstash|kibana|splunk|data warehousing|etl)\b',
        
        # Embedded Systems & IoT
        r'\b(embedded systems|iot|arduino|raspberry pi|microcontrollers|sensors|bluetooth|zigbee|mqtt|coap)\b',
        
        # Blockchain
        r'\b(blockchain|ethereum|bitcoin|smart contracts|solidity|hyperledger|dapps|cryptography)\b',
        
        # Virtualization & Containers
        r'\b(vmware|virtualbox|hyper-v|kvm|xen|docker|kubernetes|openshift|rancher)\b',
        
        # Miscellaneous
        r'\b(agile|devops|microservices|serverless|rest|graphql|api|oauth|jwt|oauth2|grpc|protobuf|websockets)\b'
    }
    
    skills = set()
    text_lower = text.lower()
    
    for pattern in tech_patterns:
        matches = re.finditer(pattern, text_lower)
        skills.update(match.group(0) for match in matches)
    
    return skills

def analyze_education_level(text):
    """Analyze education level mentioned in text"""
    education_keywords = {
        'phd': 4,
        'doctorate': 4,
        'master': 3,
        'bachelor': 2,
        'undergraduate': 2,
        'associate': 1,
        'certification': 1
    }
    
    text_lower = text.lower()
    max_level = 0
    for keyword, level in education_keywords.items():
        if keyword in text_lower:
            max_level = max(max_level, level)
    
    return max_level

def calculate_experience_match(resume_text, job_text):
    """Calculate experience match score"""
    job_exp_pattern = r'(\d+)[\+]?\s*(?:year|yr)s?\s*(?:of)?\s*experience'
    job_exp_matches = re.findall(job_exp_pattern, job_text, re.IGNORECASE)
    required_years = max([int(y) for y in job_exp_matches]) if job_exp_matches else 0
    
    resume_exp_matches = re.findall(job_exp_pattern, resume_text, re.IGNORECASE)
    candidate_years = max([int(y) for y in resume_exp_matches]) if resume_exp_matches else 0
    
    if required_years == 0:
        return 100
    return min(100, (candidate_years / required_years) * 100)

def extract_keywords(text):
    """Extract important keywords using TF-IDF"""
    vectorizer = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        stop_words=list(STOPWORDS)
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Get keywords with scores above mean
        mean_score = scores.mean()
        keywords = [
            feature_names[i] for i in range(len(scores))
            if scores[i] > mean_score
        ]
        return keywords
    except Exception as e:
        print(f"Error in keyword extraction: {str(e)}")
        return []

def analyze_resume(resume_text, job_desc_text):
    """Main function to analyze resume against job description"""
    try:
        # Tokenize texts
        processed_resume = ' '.join(tokenize_text(resume_text))
        processed_job = ' '.join(tokenize_text(job_desc_text))
        
        # Extract technical skills
        resume_tech_skills = extract_technical_skills(resume_text)
        job_tech_skills = extract_technical_skills(job_desc_text)
        
        # Calculate technical skills match
        matching_tech_skills = resume_tech_skills.intersection(job_tech_skills)
        tech_skills_score = (len(matching_tech_skills) / len(job_tech_skills) * 100) if job_tech_skills else 100
        
        # Extract and compare keywords
        job_keywords = set(extract_keywords(job_desc_text))
        resume_keywords = set(extract_keywords(resume_text))
        
        matching_keywords = job_keywords.intersection(resume_keywords)
        missing_keywords = job_keywords - resume_keywords
        keyword_match_score = (len(matching_keywords) / len(job_keywords) * 100) if job_keywords else 100
        
        # Calculate experience match
        experience_score = calculate_experience_match(resume_text, job_desc_text)
        
        # Calculate education level match
        job_edu_level = analyze_education_level(job_desc_text)
        resume_edu_level = analyze_education_level(resume_text)
        education_score = min(100, (resume_edu_level / job_edu_level * 100)) if job_edu_level > 0 else 100
        
        # Calculate overall similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_job])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
        
        # Calculate weighted overall score
        overall_score = (
            similarity_score * 0.3 +
            tech_skills_score * 0.3 +
            keyword_match_score * 0.2 +
            experience_score * 0.1 +
            education_score * 0.1
        )
        
        return {
            'overall_score': round(overall_score, 2),
            'similarity_score': round(similarity_score, 2),
            'tech_skills_score': round(tech_skills_score, 2),
            'keyword_match_score': round(keyword_match_score, 2),
            'experience_score': round(experience_score, 2),
            'education_score': round(education_score, 2),
            'matching_keywords': list(matching_keywords),
            'missing_keywords': list(missing_keywords),
            'matching_tech_skills': list(matching_tech_skills),
            'missing_tech_skills': list(job_tech_skills - resume_tech_skills)
        }
    except Exception as e:
        raise Exception(f"Error in resume analysis: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        resume_file = request.files['resume']
        job_desc_file = request.files['jobdesc']
        
        resume_text = resume_file.read().decode('utf-8')
        job_desc_text = job_desc_file.read().decode('utf-8')
        
        result = analyze_resume(resume_text, job_desc_text)
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        return jsonify(error_details), 400

if __name__ == '__main__':
    app.run(debug=True)