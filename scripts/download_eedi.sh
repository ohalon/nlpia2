# Neurips 2022 EEDI home page: https://eedi.com/projects/neurips-2022
# EEDI Competition home page: https://codalab.lisn.upsaclay.fr/competitions/5626
# Competition page link to google drive data.zip: https://1drv.ms/u/s!Ai2oY6KqGQ4Lb7SWI0hP5-geV3c?e=msNdQp
# Starting Kit "getting-data" instructions: https://codalab.lisn.upsaclay.fr/competitions/5626#learn_the_details-get_starting_kit
# 1. Task 1 Public (190 MB): https://codalab.lisn.upsaclay.fr/my/datasets/download/abe8aa94-259e-4552-a4e9-7b7d8d0bf15e
# 2. Task 1 Private (96 MB): https://codalab.lisn.upsaclay.fr/my/datasets/download/36c18dfa-00ac-4b6b-9729-042620733d23
# 3. Task 2 Public (8 MB): https://codalab.lisn.upsaclay.fr/my/datasets/download/089d7f9d-68ed-4ba3-9268-4e5450fb1507
# 4. Task 2 Private (4 MB): https://codalab.lisn.upsaclay.fr/my/datasets/download/38a8e880-474f-4465-92f7-96d15aca6634
# 5. Task 3 Public (8 MB): https://codalab.lisn.upsaclay.fr/my/datasets/download/b9df4b63-9a4b-43a4-b5a5-de184b1e7821
# 7. Task 4 Private (2 kB): https://codalab.lisn.upsaclay.fr/my/datasets/download/c116fe1f-db4c-457b-8997-a1f51b01191a

URL="https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip"
wget -m -np -c -U ~/.nlpia2-data/neurips-eedi-data.zip -- "$URL"

URL="https://drive.google.com/u/0/uc?id=1TVyGIWU1Mn3UCjjeD6bcZ57YspByUV7-&export=download"
# wget -m -np -c -U ~/.nlpia2-data/neurips-eedi-data.zip -- "$URL"
