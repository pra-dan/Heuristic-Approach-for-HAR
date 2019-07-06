#INSTALLING "PyAudio" FOR PYTHON 3.7 IS NOT THAT SIMPLE: REFER THIS :(https://stackoverflow.com/a/54998167/9625777)
import pyttsx3
import speech_recognition as sr #pip install sounddevice
import pocketsphinx             #pip install pocketsphinx

class SpeechText():

    def AskAndListen(self, question):
        engine = pyttsx3.init()
        engine.say(question)
        engine.runAndWait() 

        #LISTEN FOR REPLY
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            r.pause_threshold = 1
            audio = r.listen(source)
        '''
        #THIS WORKS WITH SOME LAG, BUT ONLINE ONLY
        try:
            print("Recognizing...")    
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")
        '''

        try:
            speech_str = r.recognize_sphinx(audio)
            print(speech_str)
            result = self.YesNoBoth(speech_str)
            return result
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))
        
    def YesNoBoth(self, speech_str):
        if(speech_str.find('yes') != -1 or speech_str.find('yeah') != -1)
            return "yes"
        if(speech_str.find('no') != -1 or speech_str.find('nah') != -1)
            return "no"


