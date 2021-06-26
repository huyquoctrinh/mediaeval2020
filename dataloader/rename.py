import os
path = "D:/Medicoeval/result1"
 # For files that are common (same names)
    os.rename(os.path.join(base_path,filename+'.jpg'),os.path.join(base_path,str(counter)+'.jpg')) # Rename jpg file
    os.rename(os.path.join(base_path,filename+'.json'),os.path.join(base_path,str(counter)+'.json')) # Rename json file
    counter += 1 # Increment counter