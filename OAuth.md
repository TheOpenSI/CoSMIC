# This document provides a Docker Compose setup to launch a complete local environment for Open WebUI, including:

PostgreSQL: user accounts and chat history storage

pgAdmin 4: web-based database management UI

Open WebUI: chat interface with local login and Microsoft SSO

## Installation

a. Prerequisites

Docker & Docker Compose installed.

A registered Azure AD application (for Microsoft SSO).

A '.env' file in this directory with your Azure credentials:

AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret

b. Docker compose guide

1. PostgreSQL

postgres:
  image: postgres:14
  container_name: postgres
  restart: always
  environment:
    POSTGRES_USER: opeuser
    POSTGRES_PASSWORD: data101
    POSTGRES_DB: openwebui
  volumes:
    - postgres-data:/var/lib/postgresql/data
  networks:
    - webnet

Purpose: A Postgres 14 instance that stores user data and chat logs.

Key settings:

    POSTGRES_USER / POSTGRES_PASSWORD: database credentials.

    POSTGRES_DB: initial database.

    Persistence: postgres-data volume.

2. pgAdmin 4

pgadmin:
  image: dpage/pgadmin4:latest
  container_name: pgadmin4
  restart: always
  environment:
    PGADMIN_DEFAULT_EMAIL: admin@local.com
    PGADMIN_DEFAULT_PASSWORD: admin
  ports:
    - '5050:80'
  volumes:
    - pgadmin-data:/var/lib/pgadmin
  networks:
    - webnet

Purpose: Browser interface at http://localhost:5050 to inspect and manage the Postgres database.

Login: admin@local.com / admin.

Persistence: pgadmin-data volume.

3. Open WebUI

open-webui:
  image: ghcr.io/open-webui/open-webui:main
  container_name: open-webui
  restart: always
  env_file:
    - .env
  environment:
    - LOG_LEVEL=debug                  # Verbose logging for debugging issues in development

    # Configuration persistence
    - ENABLE_PERSISTENT_CONFIG=false   # Disable saving previous config to ensure env vars always take effect

    # Authentication UI
    - ENABLE_LOGIN_FORM=true           # Show built-in email/password form for local accounts
    - ENABLE_OAUTH_SIGNUP=true         # Allow new users to sign up via Microsoft SSO
    - ENABLE_OAUTH_LOGIN=true          # Add "Continue with Microsoft" button on login screen

    # Microsoft OAuth credentials
    - MICROSOFT_CLIENT_ID=${AZURE_CLIENT_ID}         # From Azure AD App registration
    - MICROSOFT_CLIENT_SECRET=${AZURE_CLIENT_SECRET} # Client secret from Azure AD
    - MICROSOFT_CLIENT_TENANT_ID=${AZURE_TENANT_ID}  # "common" for personal & org accounts or your tenant ID

    # Account linking
    - OAUTH_MERGE_ACCOUNTS_BY_EMAIL=true  # Merge existing local accounts with SSO by matching email addresses

    # Database & App URL
    - DATABASE_URL=postgresql://opeuser:data101@postgres:5432/openwebui  # Connection string to Postgres
    - WEBUI_URL=http://localhost:8080    # Public-facing URL used in redirects

    # OpenID Connect discovery
    - OPENID_PROVIDER_URL=https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration
                                          # Endpoint for OIDC metadata (issuer, auth URLs, JWKS)
  ports:
    - '8080:8080'
  depends_on:
    - postgres
  networks:
    - webnet

Purpose: The main chat UI which supports both:

    Local username/password accounts.

    Microsoft SSO via Azure AD.

Key vars:

    ENABLE_LOGIN_FORM, ENABLE_OAUTH_LOGIN, ENABLE_OAUTH_SIGNUP: enable login options.

    MICROSOFT_CLIENT_*: Azure AD app credentials.

    DATABASE_URL: connection to the Postgres service.

    OPENID_PROVIDER_URL: auto-discovery endpoint for Microsoft’s OIDC.

Volumes & Network

volumes:
  postgres-data:
  pgadmin-data:

networks:
  webnet:

postgres-data: persists Postgres data.

pgadmin-data: persists pgAdmin configuration.

webnet: isolates inter-service communication.

Usage

Access pgAdmin → http://localhost:5050

Access Open WebUI → http://localhost:8080 (choose local or Microsoft login)

To stop and clean up data/containers:

docker-compose down -v

## Azure AD Application Setup

To enable Microsoft SSO, you must configure an Azure AD app. Follow these steps:

    Sign in to Azure Portal

    Visit https://portal.azure.com and authenticate with your Microsoft or Azure AD account.

    Register a New App

    From the Search Bar → App registrations 
    Click + New registration

    Name: OpenWebUI-OAuth

    Supported account types: Accounts in any organizational directory (Multitenant) and personal Microsoft accounts.

    Click Register.

    Copy IDs

    In the app’s Overview, note the Application (client) ID and Directory (tenant) ID. You will add these to your .env.

    Configure Redirect URI

    Under Manage → Authentication, click + Add a platform → Web.

    Enter: http://localhost:8080/oauth/microsoft/callback

    Click Configure.

    Set Front-Channel Logout URL

    On the same Authentication page, scroll to Front-channel logout URL.

    Enter: http://localhost:8080/auth

    Click Save.

    Create Client Secret

    Under Certificates & secrets, click + New client secret.

    Provide a description and expiration, then Add.

    Copy the Value immediately—it’s your AZURE_CLIENT_SECRET.

    Edit your .env filePlace the IDs and secret into .env alongside your docker-compose.yaml:

        AZURE_CLIENT_ID=your-client-id
        AZURE_CLIENT_SECRET=your-client-secret

    Restart your stack

        docker-compose down -v
        docker-compose up -d

Test Microsoft SSO

    Go to http://localhost:8080 and click Continue with Microsoft.

    You should be redirected to Microsoft’s login, then back as a signed-in user.