class HtmlWriter():
    def __init__(self, filename):
        self.filename = filename
        self.html_file = open(self.filename, 'w')
        self.html_file.write(
            """<!DOCTYPE html>\n""" + \
            """<html>\n<body>\n<table border="1" style="width:100%"> \n""")
        
    def add_element(self, col_dict):
        self.html_file.write('    <tr>\n')
        for key in range(len(col_dict)):
            self.html_file.write("""    <td>{}</td>\n""".format(col_dict[key]))
            
        self.html_file.write('    </tr>\n')
                
    def image_tag(self, image_path, height=240, width=320):
        return """<img src="{}" alt="{}" height={} width={}>""".format(
            image_path,image_path,height,width)

    def video_tag(self, video_path, height=240, width=320, autoplay=True):
        if autoplay:
            autoplay_str = 'autoplay loop'
        else:
            autoplay_str = ''

        tag = \
            """<video width="{}" height="{}" controls {}>""".format(width,height,autoplay_str) + \
            """    <source src="{}" type="video/mp4">""".format(video_path) + \
            """    Your browser does not support the video tag.""" + \
            """</video>"""

        return tag
    
    def colored_text(self,text,color):
        return '<span style=\"color:' + color + '\">' + text + '</span>'

    def bg_colored_text(self,text,bg_color,text_color='rgb(0,0,0)'):
        return f'<span style=\"background-color:{bg_color}; color:{text_color}\">' + text + '</span>'

    def editable_content(self,content):
        return """<div contenteditable="True">{}</div>""".format(content)
    
    def close(self):
        self.html_file.write('</table>\n</body>\n</html>')
        self.html_file.close()