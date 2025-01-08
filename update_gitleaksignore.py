import json

# Open the leak report file and read its contents
with open("leak_report", "r") as report_file:
    report = json.load(report_file)
    
    # Extract the fingerprints of each leak
    fps = [leak["Fingerprint"] + "\n" for leak in report]

# Open the .gitleaksignore file and append the fingerprints
with open(".gitleaksignore", "a") as ignore_file:
    ignore_file.writelines(fps)
