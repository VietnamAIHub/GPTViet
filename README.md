
<h1 align="center">
  <span> GPTViet - Advancing Foundation Models</span>
</h1>

This project aims to develop a multilingual foundation model both language and multimodal capabilities. The objective is to enhance an existing Open-source English based model, optimizing it for the Vietnamese and others language.

<h3 align="center">
  <span> GPTViet - Target Development</span>
</h3>

<div align="center">
     <img width="auto" height="400px" src="./GPTViet.png"/>
</div>

## 💡 Get help - [Q&A](https://github.com/TranNhiem/Vietnamese_LLMs/discussions) or [Discord 💬](https://discord.gg/BC8Mqq8qYn)



## 1. Roadmap Development of GPTViet's Language Foundation Model: 

<div align="center">
     <img width="auto" height="400px" src="./GPTViet_llm.png"/>
</div>

## Demo Language Model: 

<h3 align="center">
  <span> Watch/Xem GPTViet Assistant Demo </span>
</h3>
<div align="center">
  <a href="https://youtu.be/B0bDwsAli_k">
    <img src="https://img.youtube.com/vi/B0bDwsAli_k/0.jpg" alt="Watch the video" width="500px" height="auto">
  </a>
</div>


+ [**GPTViet 8B Demo Chat & Websearch Integration**](http://140.115.53.106:8888/)
+ [**GPTViet 70B Demo Coming Soon**]()
+ [**GPTViet Document Chat Demo Coming Soon**]()

## Performance Benchmarks on Multiple Task: 

+ Comprehensive and Advanced Vietnamese Benchmark for Language Model
  
| Benchmark Category                     | Benchmark Task Description                       | Metric     | Number of Samples |
|----------------------------------------|--------------------------------------------------|------------|-------------------|
| **<span style="font-size:13px">General Knowledge</span>**                  |                                                  |            |                   |
| <span style="font-size:11px"> Vietnamese Exam (Từ lớp 6->12,& THPT)</span>           | <span style="font-size:11px">Đánh giá Tổng hợp bộ câu hỏi trắc nghiệm cho các bộ môn (Toán, Lý, Hoá, Anh, Sinh vv..) dựa trên các bộ đề thi ở Việt nam từ lớp 6 đến lớp 12 và bao gồm thi Trung học phổ thông quốc gia </span>  |    Prefix Match (Accuracy)         |      33, 000           |
| <span style="font-size:11px"> VMLU Vietnamese Multitask Language Understanding</span>                        | <span style="font-size:11px"> Đánh giá dựa trên câu hỏi trắc nghiệm bao gồm 58 chủ đề khác nhau, được phân bố qua bốn lĩnh vực chính: STEM, Nhân văn, Khoa học Xã hội, và hơn thế nữa. Nó bao trùm nhiều cấp độ khó khác nhau, từ trình độ cơ bản đến chuyên môn nâng cao, thách thức các mô hình nền tảng trong cả kiến thức chung và giải quyết vấn đề phức tạp.</span>   |     Prefix Match (Accuracy)     |       10,880             |
| **<span style="font-size:13px">Summarization (Short & Long)</span>**       |                                                  |            |                   |
| <span style="font-size:11px">BìnhNews (ROUGH_1,2)</span>                   | <span style="font-size:11px">Tóm tắt các ý chính quan trọng của một đoạn văn</span>       | <span style="font-size:11px">ROUGH_1,2 & Embedding Similarity</span>  |                |
| <span style="font-size:11px">VietNews (ROUGH_1,2)</span>                   | <span style="font-size:11px">Tóm tắt Văn bản dựa trên một câu chính để mô tả nội dung của đoạn văn</span> | <span style="font-size:11px">ROUGH_1,2 Embedding Similarity</span>  |         2,800             |
| **<span style="font-size:13px">Translation</span>**                        |                                                  |            |                   |
| <span style="font-size:11px"> Flore 101 (EN2Vi & Vi2EN) </span>            | <span style="font-size:11px"> Bảng đánh giá Flores-101 bao gồm 3001 câu được trích xuất từ Wikipedia tiếng Anh sang các Ngôn ngữ khác và bao gồm một loạt các chủ đề và lĩnh vực khác nhau.</span> | <span style="font-size:11px">(BLEU)& Embedding Similarity </span>       |  3001            |
| **<span style="font-size:13px">Human Benchmark</span>**                    |                                                  |            |                   |
| <span style="font-size:11px">SeaBench (Realworld_Test)</span>              | <span style="font-size:11px">đánh giá các Mô hình Ngôn ngữ LLMs như các trợ lý hữu ích, bao gồm các loại hướng dẫn đa dạng để đánh giá các mô hình, như mô tả sau đây: Giải quyết vấn đề: Đánh giá 1. khả năng xử lý ngôn ngữ tự nhiên thông qua các nhiệm vụ như tóm tắt và dịch. 2. Suy luận toán học: Đánh giá kỹ năng suy luận toán học và logic. 3.Dữ liệu hướng dẫn tổng quát: Kiểm tra kiến thức tổng quát và kỹ năng viết, bao gồm tạo ra các ý tưởng sáng tạo và phản hồi yêu cầu của người dùng. 4. NaturalQA: Phân tích phản ứng với ngôn ngữ tự nhiên và ngữ cảnh địa phương từ các truy vấn thực tế của người dùng. 5. An toàn: Đảm bảo sự hiểu biết về các quy tắc và quy định an toàn, bao gồm ngữ cảnh.</span> |       Human/GPT4 as a Judge     |          130         |


+ Access Benchmark Performance Setting

<!-- 
| LLM Model                 | General Knowledge | Summarization (Short & Long) | Translation | Human Benchmark |
|---------------------------|-------------------|------------------------------|-------------|-----------------|
|                           | Việt Exam <br> VMLU (Val+Test) | BìnhNews (ROUGH_1,2) <br> VietNews (ROUGH_1,2) | EN2Vi (BLEU) <br> Vi2EN (BLEU) | SeaBench (Realworld_Test) |
| GPTViet 8B (Small) | 38.96%                 | 47.95%                            | %           | %               |
| GPTViet 70 (Large) | 53.65%                 | 69.72%                            | %           | %               |
| Llama 3 70 Instruct (Meta)| 50.90%                 | 51.38%                            | %           | %               |
| Llama 3 8B Instruct (Meta)| 31.94%                 | %                            | %           | %               |
| Vistral (Mistral 7B)      | %                 | %                            | %           | %               |
| SeaLLM v2.5 (Llama2-7B)   | %                 | %                            | %           | %               |
| GPT-3.5                   | %                 | %                            | %           | %               |
| GPT-4 (Turbo)             | %                 | %                            | %           | %               |
-->

## Multi Domains – General Knowledge Understanding Viet Exam (Lớp 6 -> THPT) & VLMU Benchmark

- **Viet Exam:** Collection of 10k Questions and answers across all Subjects from Vietnamese Exam from 6 to 12 Grades.
- **The VMLU Dataset:** Comprises 58 distinct subjects. These subjects are systematically categorized into four primary domains: STEM, Humanities, Social Sciences, and a broad category designated as 'Others.'

| LLM Model                      | Việt Exam (10,000 Questions) Prefix Match Accuracy | VLMU (Val+Dev) (1048 Questions) Prefix Match Accuracy |
|--------------------------------|----------------------------------------------------|-----------------------------------------------------|
| GPTViet 8B (2024-05-Version)   | 39.28%                                             | 48.04%                                              |
| GPTViet 70 (2024-06-Version)   | 53.80%                                             | 70.20%                                              |
| Llama 3 70 Instruct (Meta)     | 50.50%                                             | 51.58%                                              |
| Llama 3 8B Instruct (Meta)     | 32.07%                                             | 23.59%                                              |
| GPT-3.5 (OpenAI)               | --%                                                | --%                                                 |
| GPT-4 (Turbo) (OpenAI)         | --%                                                | --%                                                 |

## Vietnamese-English Translation Flores 101 Benchmark

Flores-101 evaluation benchmark, consisting of 3001 sentences extracted from English Wikipedia to Other Languages and covering a variety of different topics and domains.

| LLM Model                      | Translation Vietnamese--English |          |          | Translation English--Vietnamese |          |          |
|--------------------------------|---------------------------------|----------|----------|---------------------------------|----------|----------|
|                                | BLEU Score                      | ROUGH-1  | ROUGH-2  | BLEU Score                      | ROUGH-1  | ROUGH-2  |
| GPTViet 8B (2024-05-Version)   | 21.8%                           | 56.03%   | 31.42%   | 26.0%                           | 59.65%   | 36.14%   |
| GPTViet 70 (2024-06-Version)   | 27.6%                           | 61.15%   | 37.39%   | 33.3%                           | 67.08%   | 43.97%   |
| Llama 3 70 Instruct (Meta)     | 25.0%                           | 58.14%   | 35.02%   | 33.0%                           | 66.56%   | 43.63%   |
| Llama 3 8B Instruct (Meta)     | 19.3%                           | 51.62%   | 28.15%   | 9.8%                            | 24.16%   | 13.92%   |
| GPT-3.5 (OpenAI)               | 28.1%                           | 62.05%   | 38.14%   | 34.0%                           | 67.0%    | 44.63%   |
| GPT-4 (Turbo) (OpenAI)         | --%                             | --%      | --%      | --%                             | --%      | --%      |



## Download & Get Latest Version GPTViet: 
+ Assistant Language Model (GPTViet Beta 1.0  Small & Large GPTViet Assistant )
    - Websearch Assistant Model
    - ChatDocument Assistant Model
    - Custom Service Assistant Model
      
+ Translation Langauge Model from GPTViet Branch  Beta 1.0 Small & Large
  - VietTranslate English->Vietnamese , Vietnamese -> English

+ Medical Langauge Model from GPTViet Branch  Beta 1.0 Small & Large
  - VietMed trợ lý sức khoẻ
   
Bạn có thể kết nối trực tiếp với Trần Nhiệm [email]: tvnhiemhcmus@gmail.com


## Giúp Đỡ (How You can HELP)
1. Nhằm để hỗ trợ tài chính Nhóm xin nhận làm các dự án của công ty tư nhân, các tổ chức nghiên cứu, hoặc cá nhân.
2. Bạn có thể hổ trợ về tài nguyên như máy chủ server hoặc các tài nguyên khác.
  - Dự án hiện rất cần các nguồn tài trợ tài nguyên GPUs để có thể tiến hành quá trình huấn luyện (Training Model).
  - Bạn có thể kết nối trực tiếp với Trần Nhiệm [email]: tvnhiemhcmus@gmail.com. Hoặc có thể chat trực tiếp ở: [LinkedIn](https://www.linkedin.com/in/tran-nhiem-ab1851125/) [Facebook](https://www.facebook.com/jean.tran.336). [X](https://twitter.com/TranRick2). [Zalo +886 934 311 751]()
