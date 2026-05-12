# Implementing Microsoft OAuth in Your Application  

## Azure AD Application Setup  

To enable Microsoft SSO, follow these steps to configure an Azure AD app:  

### 1. Sign in to Azure Portal  
Visit [Azure Portal](https://portal.azure.com) and authenticate with your Microsoft or Azure AD account.  

### 2. Register a New App  
- Navigate to **App registrations** using the search bar.  
- Click **+ New registration**.  
- Fill in the following details:
    - **Name**: cosmic-openwebui  
    - **Supported account types**: Accounts in any organizational directory (Any Microsoft Entra ID tenant - Multitenant) and personal Microsoft accounts (e.g. Skype, Xbox)  
    > **Note**: You can change these details according to your preference.  
    - Under **Redirect URI**, select **Web** as the platform and add the following URI:  
        ```
        <open-webui>/oauth/google/callback  
        ```
    - In our case, use the URI:  
        ```
        http://localhost:8080/oauth/microsoft/callback  
        ```
    - Click **Register**.  

For more detailed instructions, you can also refer to the official Microsoft documentation on registering an app: [Quickstart: Register an application](https://learn.microsoft.com/en-us/entra/identity-platform/quickstart-register-app).

### 3. Copy IDs  
- In the app’s **Overview**, note the **Application (client) ID** and **Directory (tenant) ID**.  
- Add these values to your `.env` file.  

### 4. Create Client Secret  

- Navigate to **Certificates & secrets** under the **Manage** section of your app.  
- Select **Client secrets** and click **+ New client secret**.  
- Add a description and choose an expiration period for the client secret.  
- Click **Add** to create the client secret.  
- Copy the **Value** of the client secret and add this to your `.env` file.

### 7. Edit Your `.env` File  
Place the IDs and secret into your `.env` file alongside your `docker-compose.yaml`:  
```env  
AZURE_CLIENT_ID=<your-client-id>  
AZURE_CLIENT_SECRET=<your-client-secret>
AZURE_TENANT_ID=<your-tenant-id>
```  
### 8. Start Your Application  

- After completing the setup, start your application follwing the [README](README.md).
- Your application is now configured to use Microsoft OAuth!

## Configuring Other OAuth Services  

Open WebUI supports federated authentication with various OAuth providers. To configure additional OAuth services, follow the steps outlined in the [Open WebUI Federated Authentication Support documentation](https://docs.openwebui.com/features/sso).  