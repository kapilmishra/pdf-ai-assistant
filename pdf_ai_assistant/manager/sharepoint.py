import json
import os
from typing import Optional

from office365.runtime.auth.authentication_context import AuthenticationContext  
from office365.sharepoint.client_context import ClientContext  
from office365.runtime.auth.client_credential import ClientCredential
from office365.sharepoint.files.file import File  


class SharePointAPI:
    siteURL = os.environ['SHAREPOINT_SITE_URL']
    clientID =  os.environ['SHAREPOINT_CLIENT_ID']
    clientSecret = os.environ['SHAREPOINT_CLIENT_SECRET']
    
    username =  os.environ['SHAREPOINT_USER']
    password = os.environ['SHAREPOINT_PASSWORD']

    def __init__(self):
        self.listFiles("test")
    
    def listFiles(self, path: str):
        context = self.getContext()
        web = context.web  
        context.load(web)  
        context.execute_query()  
        print("Web site title: {0}".format(web.properties['Title'])) 
        
    #def downloadFile(self, path: str):
        

    def getContext(self):
       client_credentials = ClientCredential(self.clientID,self.clientSecret)
       return ClientContext(self.siteURL).with_credentials(client_credentials)
       
    
SharePointAPI();