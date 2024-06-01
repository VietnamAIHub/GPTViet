import subprocess
import os
import requests
import json


### Helper function to format the chat messages and Print out the response

def apply_chat_template(example):
    messages = example["messages"]
    formatted_chat = ""

    eos_token = "<|eot_id|>"  
    if messages and messages[0]["role"] == "system":
        # Update the content of the first system message
        previous_content =  messages[0]["content"] #"<|begin_of_text|>system<|start_header_id|>\n"+
        messages[0]["content"] = "<|begin_of_text|>system<|start_header_id|>\n" + previous_content +eos_token
    ## Add an empty system message if there is no initial system message
    elif messages and messages[0]["role"] != "system":
        # Insert a new system message at the beginning if there isn't one
        messages.insert(0, {"role": "system", "content": "<|begin_of_text|>system<|start_header_id|>\nYou are GPTViet created by VietnamAIHub. Designed to help users find detailed and comprehensive information. Always aim to provide answers in such a manner that users don't need to search elsewhere for clarity.<|eot_id|>"})
        
    # Define your end-of-sentence token here
    eos_token_="<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    # Loop through the messages to apply the template
    for i, message in enumerate(messages):
        role = message['role']
        
        if role =="user":
            content = "<|start_header_id|>user<|end_header_id|>\n\n" + message['content'] + eos_token_
            formatted_chat += f'{content}'
        
        elif  role =="assistant":
            content = message['content'] + eos_token +"\n"
            formatted_chat += f'\n{content}' 
        else :
            content = message['content']
            formatted_chat += f'{content}' 

            
    return formatted_chat

def print_response(response):
    if response.status_code == 200:
        all_content = []
        # Iterate over the response stream line by line
        for line in response.iter_lines():
            if line:
                try:
                    response_json = json.loads(line.decode('utf-8'))
                    content = response_json.get("response", "")
                    if content:
                        all_content.append(content)
                        print(content, end='', flush=True)  # Print content as it arrives
                except json.JSONDecodeError as e:
                    print("Error parsing JSON:", e)

        print()
    else:
        print("Request failed with status code:", response.status_code)
    return all_content

DEFAULT_SYSTEM_PROMPT = """\nYou are GPTViet created by VietnamAIHub. Designed to help users find detailed and comprehensive information. Always aim to provide answers in such a manner that users don't need to search elsewhere for clarity."""
messages = {
"messages": [
    {"role": "system", "content": f"{DEFAULT_SYSTEM_PROMPT}"}
]
}
## Optional 
# Append the user message
messages["messages"].append({"role": "user", "content": "chào bạn cho hỏi 1+1=?"})
# Append the assistant response
messages["messages"].append({"role": "assistant", "content": "Câu trả lời cho bài toán đơn giản trên là  1+1 = 2."})  
## Now user Question or Input  
messages["messages"].append({"role": "user", "content": f"cho tôi biết bạn là ai?"})  

input_prompt = apply_chat_template(messages)

url = "http://40.84.133.133:8889/api/generate"
payload = {
    "model": "GPTViet_2024_04_version",
    "prompt": input_prompt,
    "stream": True, 
    "options": {
        # "seed": 123,
        "temperature": 0.2 
        # "top_k": 20,
        # "top_p": 0.9,
        # "tfs_z": 0.5,
        # "typical_p": 0.7,
        # "repeat_last_n": 33,
        # "repeat_penalty": 1.2,
        # "presence_penalty": 1.5,
        # "frequency_penalty": 1.0,
    }
}
response = requests.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, stream=True)
respone_output=print_response(response)

