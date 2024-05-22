FROM 0_wsclean_base:latest


#####################################################################
## BUILD DSA2000_CAL
#####################################################################

WORKDIR /dsa/code
COPY dsa2000_cal/requirements.txt dsa2000_cal/requirements.txt
RUN pip install -r dsa2000_cal/requirements.txt
COPY dsa2000_cal dsa2000_cal
RUN pip install ./dsa2000_cal