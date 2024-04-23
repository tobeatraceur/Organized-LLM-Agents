import multiprocessing
import subprocess
import os
import gradio as gr
from autogen import oai
import json

def postprocess(s):
    """
        for markdown compatibility
    """
    s = s.replace('<', '\<')
    s = s.replace('>', '\>')
    return s

def gradio_server(conn, port):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def init():
            msg = str(conn.recv())
            msg =  postprocess(msg)
            return [(None, msg)]

        chatbot.value  = init()
        clear.click(init, outputs = [chatbot])


        def respond(message, chat_history):
            conn.send(message)
            reply = conn.recv()
            reply = str(reply)
            reply = postprocess(reply)
            chat_history.append((message, reply))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

    demo.queue()
    demo.launch(server_port = port, share=True)



class ChatCompletionManager:
    def __init__(self):
        self.human_agents = {}
        self.next_port = 7861

    def new_human_agent(self):
        parent_conn, child_conn = multiprocessing.Pipe()

        # get a port
        port = self.next_port
        self.next_port += 1

        # Start the other script as a subprocess
        p_gradio = multiprocessing.Process(target=gradio_server, args=(child_conn, port))
        p_gradio.start()

        self.human_agents[port] = (parent_conn, child_conn, p_gradio)
        return port

    def close(self):
        for port, item in self.human_agent_dict.items():
            # Close the parent connection
            item[2].join()
        for port, item in self.human_agent_dict.items():
            item[0].close()

    def create(self, messages, config_list=None, *args, **kwargs):
        if kwargs.get('human_agent') is None:
            return oai.ChatCompletion.create(messages=messages, config_list=config_list, request_timeout=600, *args, **kwargs)

        # human player
        else:
            port = kwargs['human_agent']
            parent_conn, child_conn, _ = self.human_agents[port]
            str_message = ""
            for i in range(len(messages)):
                str_message += f"{messages[i]['role']} :" + messages[i]['content'] + '\n'

            parent_conn.send(str_message)
            str_result = parent_conn.recv()
            print("str_result:", str_result)

            #TODO postprocessing is needed

            if '{' not in str_result:
                str_result = '{"action": "' + str_result + '", "thoughts": ""}'

            # json_result = json.loads(str_result)
            # if json_result.get('thoughts') is None:
            #     json_result['thoughts'] = ""

            # str_result = json.dumps(json_result)

            result = {
                'usage': {
                    'prompt_tokens': 9999,
                    'completion_tokens': 9999,
                },  
                'choices': [ { 'message': {'content': str_result, 'role': 'assistant'} } ]
            }
            return result


oai_wrapper = ChatCompletionManager()

if __name__ == "__main__":
    human = oai_wrapper.new_human_agent()
    res = oai_wrapper.create(messages="Test", human_agent = human)
    print(res)
