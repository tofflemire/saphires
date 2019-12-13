import os
import re

def directory_iterator_map(action_fn,boolean_fn,path_of_directory):
    #iterate through all items in the directory and do something to them
    for entry in os.scandir(path_of_directory):
        if boolean_fn(entry.path):
            scrape_comments(entry.path)

def scrape_comments(file_path):
    print(file_path+"-------------------")
    file = open(file_path,"r")
    #go through the file
    scrapers = []
    for line in file:
        if line.startswith("#skip"):
            line = next(file)
            continue
        if line.startswith("def"): #this means we have a function, and expect comments
            method_header = ""
            while(not re.match("(\s|\t)*('|\")('|\")('|\")",line)): #paginating to start of comments
                method_header += line
                line = next(file)

            scraper = comment_scraper(file, method_header)#create our scraper object at the correct pos
            scrapers.append(scraper)
    file.close()

    #now we have all of our scraped comments
    file_name = file_path.split("/")[-1]#the file name is the text after the last stroke
    file_name = file_name.split(".")[0]#stripping off any file extensions
    print(file_name)
    write_file = open("docs/source/"+file_name+".rst", "w")
    for scraper in scrapers:
        write_file.write(scraper.get_total_documentation())
        write_file.write("\n\n")


class comment_scraper():

    def get_total_documentation(self):
        return self.header + self.total_comment

    def process_comments(self,file_object):
        # We have a file_object, starting at start of the comments
        line = next(file_object)
        while (not re.match("(\s|\t)*('|\")('|\")('|\")",line)):
            #this if and for loop finds all the parameter and returns headers,
            #then surrounds them in the right number of *'s for rst formatting'
            #the right number is the same as the number of characters in the word
            if re.match("(\s|\t)*(Parameters|Returns)",line) :
                line = line.strip()
                subtitle_surrounder = ""
                for item in line:
                    subtitle_surrounder+="*"
                self.total_comment+=subtitle_surrounder+"\n"+line+"\n"+subtitle_surrounder+"\n"

                next(file_object)#to skip the underscoring currently under returns and parameters
            else:
                self.total_comment +=line.strip() +"\n"
            line = next(file_object)

    def make_header_with_rst(self, header):
        #this is to add the rst formatting to the header
        #the replace here is to make headers in rst not crazy
        header = header.replace("\n","").replace(" ","").replace("\t","")[3:]
        title_marker = ""
        for item in header:
            title_marker +="="
        title_marker +="\n"
        return title_marker + header+"\n"+title_marker

    def __init__(self,file_object, header):
        self.total_comment = ""
        self.header = self.make_header_with_rst(header)
        self.process_comments(file_object)


if __name__ == "__main__":
    utils = lambda a: a.endswith("utils.py")
    py_file_boolean = lambda a : a.endswith(".py") and not a.endswith("__init__.py")
    directory_iterator_map(scrape_comments, py_file_boolean,"saphires")
