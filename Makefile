include .env

code-to-remote:
	scp ./*.py $(REMOTE)

model-to-local:
	scp $(REMOTE)/out/model ./out/model
