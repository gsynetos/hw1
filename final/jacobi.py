#-------------------------------
#Import Library & Settings
#-------------------------------
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------
#Set the Grid
#-------------------------------
L, H = 1,1 #Size of the grid (Length,Height)
IM,JM =4,4 #Number of columns & rows
Dx,Dy = L/(IM-1) , H/(JM-1)
a = Dx/Dy 
f = 4 # Η τιμή απο τη εξίσωση Poison που καλούμαστε να λύσουμε

# Set array size and set the interior value with v_init
v_init = 0 #initial guess for interior nodes
v = np.empty((JM, IM))
v_new = np.empty((JM, IM))
A= np.empty((JM, IM))
v.fill(v_init)

#-------------------------------
#Analytical Solution
#-------------------------------
u = np.empty((JM, IM))
for i in range(JM):
    for j in range(IM):
        u[i, j] = (j*Dx)**2 + (i*Dy)**2
uk = u.flatten()

#-------------------------------
#Set Boundary conditions
#-------------------------------
#Top Edge
bc_top = np.empty(IM)
for i in range(IM):
    bc_top[i] = (i*Dx)**2 + ((JM-1)*Dy)**2
v[(JM-1):, :] = bc_top
  
#Bottom Edge
bc_bottom = np.empty(IM)
for i in range(IM):
    bc_bottom[i] = (i*Dx)**2
v[:1, :] = bc_bottom

#Right Edge
bc_right = np.empty(JM)
for i in range(JM):
    bc_right[i] = (i*Dy)**2 + ((IM-1)*Dx)**2
bc_right = bc_right[np.newaxis, :].T
v[:, (IM-1):] = bc_right

#Left Edge
bc_left = np.empty(JM)
for i in range(0, JM):
    bc_left[i] = (i*Dy)**2
bc_left = bc_left[np.newaxis, :].T
v[:, :1] = bc_left 

vk = v.flatten() # vk einai i metavliti me arithmisi kordoni i opoia perilambanei tis oriakes times

#-------------------------------
#Find the Boundary Nodes
#-------------------------------
BN = list()
#Nodes located on Bottom Side
for k in range (IM):
    BN.append(k)
#Nodes located on Top Side
for k in range ((IM*(JM-1)),IM*JM):
    BN.append(k)
#Nodes located on the Left Side excluding first and last node
for i in range (1,JM-1):
    k = i*(IM)
    BN.append(k)
#Nodes located on the Right Side excluding first and last node
for i in range (1,JM-1):
    k = (i+1)*IM-1
    BN.append(k)
BN = sorted(BN)

#-------------------------------
#Vector b
#-------------------------------

b = np.empty(IM*JM)
b[:] = vk
for k in range (IM*JM):
    if k not in BN:
        b[k]=4

#-------------------------------
#Matrix A
#-------------------------------
dim = (IM*JM,IM*JM)
A = np.zeros(dim)

#Boundary Nodes A[i,i]=1
for index in BN:
    A[index,index] = 1
    
#Inner Nodes
for k in range (IM*JM):
    if k not in BN:
        A[k,k]= -2*(Dx**2+Dy**2)/((Dx**2) * (Dy**2))
        vertical_neighbours = [k+IM,k-IM]
        for v_neighbour in vertical_neighbours:
            A[k,v_neighbour]= Dx**2/((Dx**2) * (Dy**2))
        horizontal_neighbours = [k+1,k-1]
        for h_neighbour in horizontal_neighbours:
            A[k,h_neighbour]= Dy**2/((Dx**2) * (Dy**2))


#-------------------------------
#Matrix Decomposition D-(L+U)
#-------------------------------
Diag = np.diag(np.diag(A))
Lower = np.tril(A,-1)
Lower = -Lower
Upper = np.triu(A,1)
Upper = -Upper

#-------------------------------
#Conditional Number
#-------------------------------
def conditional_number(matrix):
    inv_matrix = np.linalg.inv(matrix)
    cond_a = np.linalg.norm(matrix) * np.linalg.norm(inv_matrix)
    return cond_a

cond_a = conditional_number(A)

Jacobi = "yes"
if Jacobi == "yes":
    #-------------------------------
    #Jacobi Method
    #-------------------------------
    v_newk = vk[:]
    counter= list()
    residual = list()
    e = list()
    resid = list()
    error = list()
    res=1
    iteration = 0
    while res > 10**-6 and iteration in range(0, 100) :
        restot = 0
        vk[:]=v_newk

        for k in range (IM*JM):
            if k not in BN:
                v_newk[k] = (Upper[k,k+IM]*vk[k+IM] + Lower[k,k-IM]*vk[k-IM] + Upper[k,k+1]*vk[k+1] + Lower[k,k-1]*vk[k-1] + b[k] )/ Diag[k,k]


        res = np.linalg.norm(b - np.dot(A,v_newk)) #calculate the residual
        

        #Convergence data
        counter.append(iteration)
        residual.append(res)
        iteration += 1

        #Error
        e = (uk - v_newk)
        error.append(np.linalg.norm(e))
        
    print("After",iteration," iterations,",
         "the residual on the final iteration is ",res)
    print("The vector {v} at last iteration was calculated to be=\n",v_newk.reshape(JM,IM))
    print("The is error at last iteration = \n",np.linalg.norm(e))

        
v_new = v_newk.reshape(JM,IM)

  

#-------------------------------
#Plot Contour Lines
#-------------------------------       
# Create Grid
X, Y = np.meshgrid(np.arange(0, L, L/IM), np.arange(0, H, H/JM))
# Configure the contour
cp = plt.contour(X, Y, v_new)
plt.clabel(cp, inlie=True, fontsie=10)
plt.title("Contour Lines")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
# Set Colorbar
plt.colorbar()
# Show the result in the plot window
plt.show()   

#-------------------------------
#Plot Convergence
#-------------------------------  
plt.plot(counter, residual)
plt.yscale('symlog')
plt.title('Convergence')
plt.xlabel('iterations')
plt.ylabel('residual log_10')
plt.grid(True)
plt.show()

#-------------------------------
#Plot Error
#-------------------------------  
plt.plot(counter, error)
plt.title('error')
plt.yscale('symlog')
plt.xlabel('iterations')
plt.ylabel('error')
plt.grid(True)
plt.show()