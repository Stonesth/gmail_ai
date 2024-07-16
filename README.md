# gmail_ai

python -m venv venv
source venv/bin/activate # Sur Unix/Linux/macOS
myenv\Scripts\activate # Sur Windows

pip install google-auth
pip install google-auth-oauthlib
pip install google-auth-httplib2
pip install google-api-python-client
pip install beautifulsoup4
pip install transformers
pip install torch torchvision
pip install tensorflow
pip install tf-keras
pip install transformers
pip install langdetect
pip install googletrans
pip install googletrans==4.0.0-rc1

# Pour gérer les avertissements de tokenizers :

export TOKENIZERS_PARALLELISM=false

# Need to be run before running your Python script

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Étapes pour configurer l'accès à l'API Gmail :

## Allez sur la console des développeurs Google :

Ouvrez votre navigateur et accédez à la console des développeurs Google.
https://console.cloud.google.com/cloud-resource-manager

## Créez un projet ou utilisez un projet existant :

Si vous n'avez pas de projet existant, cliquez sur le menu déroulant en haut à gauche (à côté du logo Google Cloud Platform) et sélectionnez Nouveau projet.
Donnez un nom à votre projet et cliquez sur Créer. Si vous avez déjà un projet, sélectionnez-le dans le menu déroulant.

## Activez l'API Gmail pour ce projet :

Une fois le projet sélectionné, allez dans le menu de navigation de gauche et cliquez sur API et services > Bibliothèque.
Ou plus simplement dans la barre de recherche taper API et services

Recherchez "Gmail API" dans la barre de recherche et sélectionnez Gmail API.
Cliquez sur Activer pour activer l'API Gmail pour votre projet.

## Configurez les informations d'identification OAuth 2.0 :

- Après avoir activé l'API Gmail, allez dans le menu de navigation de gauche et cliquez sur API et services > Identifiants.

- Cliquez sur le bouton Créer des identifiants et sélectionnez ID client OAuth.

- Vous serez invité à configurer l'écran de consentement OAuth si ce n'est pas déjà fait. <b>Cliquez sur Configurer l'écran de consentement.</b>

  - Choisissez le type d'utilisateur (Interne ou Externe) et remplissez les informations nécessaires. Pour une utilisation personnelle, <b>Interne</b> est généralement suffisant.
  - Ajoutez les informations nécessaires, comme le nom de l'application, et cliquez sur <b>Enregistrer</b>.

- Une fois l'écran de consentement configuré, revenez à <b>API et services > Identifiants </b> et cliquez sur <b>Créer des identifiants > ID client OAuth</b>.

- Sélectionnez <b>Application Web</b> comme type d'application.

- Donnez un nom à vos identifiants, et dans les champs URI de redirection autorisés, ajoutez <b>http://localhost</b>.

- Cliquez sur Créer. Un fichier credentials.json sera généré.
