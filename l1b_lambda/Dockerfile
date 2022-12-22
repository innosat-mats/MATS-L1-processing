FROM public.ecr.aws/lambda/python:3.9

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY ./level1b/requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Install Mats L1 Processing (requires wheel built and copied to build tree)

COPY mats_l1_processing-0.0.0-py2.py3-none-any.whl .
RUN  pip3 install mats_l1_processing-0.0.0-py2.py3-none-any.whl --target "${LAMBDA_TASK_ROOT}"

# Copy instrument data

COPY ./calibration_data /

# Copy function code
COPY ./level1b/handlers/level1b.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "level1b.lambda_handler" ] 
