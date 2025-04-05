import numpy as np

# set decimal
np.set_printoptions(precision=4, suppress=True)

print("------------------START------------------")

Data = open("DataLess.txt") ######################################################## Import the data
List = []
for i in Data:
    x = i.split("\t")
    List.append(x)

Data.close()

# transform all values in list to a type = float
for i in range(1, len(List) ) :   #(1:314) 313
    for j in range(0,len(List[0])):   #(0,18) 19
        List[i][j] = float( List[i][j] )
        #print( List[i][j] )

# number of columns (exclude ID, 1st col, but include dependent variable Y)
Columns = len(List[0])-1
# number of rows, excluding 1st row (names)
Rows = len(List) -1

#NUMPY
'''
# X = [ 1 X1 X2.... Xn , 2 ...]
# Y = [y1 , y2 , ... , yn]          
# b = [b0 , b1 , ... , bn]
'''

#define an empty list (and later convert to matrix)
X=[]

#for the x matrix all 1st values are 1
for i in range(0,len(List)-1):
    X.append([1.0])

for i in range( 1, len(List)):
#    Xmatrix[i].append( List[i][2:] )    
    #for j , take only the independent variables
    for j in range(2, len(List[0])):
        X[i-1].append( List[i][j] )

# X -> matrix! 
Xmatrix = np.matrix(X)

#Same stuff for Y
Y = []
for i in range(1,len(List)):
    Y.append([List[i][1]])

Ymatrix = np.matrix(Y)
B = 1
#Calculate parameters (b0,b1,b2,...,b3) for Multiple Linear regression equation 

# B = (X'X)^(-1)*X'*Y 
# X'
Xtranspose = np.matrix.transpose(Xmatrix)
# X'X
XtransposeX = np.dot(Xtranspose,Xmatrix)
# (X'X)**-1
XtXinv = np.linalg.inv(XtransposeX)
# ((X'X)**-1)*X'
XtXinvX = np.dot(XtXinv,Xtranspose)
# ((X'X)**-1)*X'*Y = B
XtXinvXY = np.dot(XtXinvX,Ymatrix)
Bmatrix = (np.linalg.inv( (np.matrix.transpose(Xmatrix)*Xmatrix) ))*(np.matrix.transpose(Xmatrix))*Ymatrix

#Check
if XtXinvX.all() == Bmatrix.all():
    pass
    #print("indeed")

'''
#Errors E = X*B - Y
Ematrix = Xmatrix*Bmatrix - Ymatrix

#DOF (degrees of freedom) = f (313 - 18 = 295)
f = len(Xmatrix)-len(Bmatrix)

#Standard Error
Sigma0 = ((float(np.dot((np.matrix.transpose(Ematrix)),Ematrix)))/f)**0.5

#pinakas Symmetavlitotitas agnoston Edd(matrix)

Eddmatrix = (Sigma0**2)*XtXinv

#typika sfalmata ton parametron
EddDiagonalmatrix = Eddmatrix.diagonal()

EddDiagSquared = []
for i in range(len(Bmatrix)):
    EddDiagSquared.append((EddDiagonalmatrix[0,i])**2)
EddDiagSquaredmatrix = np.matrix(EddDiagSquared)

'''


# y = List[2]
# Set apart the Actual Y values
ActualY=[]
for i in range(1,len(List)):
    ActualY.append(List[i][1])

# calculate predicted Y values
PredictedYmatrix = Xmatrix.dot(Bmatrix)
PredictedYarray = np.squeeze(np.asarray(PredictedYmatrix))
PredictedY = []
for i in range(len(PredictedYarray)):
    j = PredictedYarray[i]
    PredictedY.append(round(float(np.squeeze(j)),4))

# Mean value of Actual Y  (sum = 257591, mean = 823)
Count = 0
for i in range(1,len(List)):
    Count += List[i][1]
MeanY = Count/Rows

#Initialize the headers of the new list (PreResidual)
PreResidual = [ [List[0][0], List[0][1], "MeanY", "Predicted", "SSTi","SSRi","SSEi"  ] ]
# calculate SST,SSR & SSE
'''
SST = Sum of Squares Total (Total in spss)
SSR = Sum of Squares Regression (Regression in spss)
SSE = Sum of Squares Error (Residual in spss)
Formulas:
SST = sum((yi - ymean)**2)
SSR = sum((ypredicted - ymean)**2)
SSE = sum((ypredicted - yi)**2)
'''
for i in range(1, len(List)):
    SST = (List[i][1] - MeanY)**2
    SSR = (PredictedY[i-1] - MeanY)**2
    SSE = (PredictedY[i-1] - List[i][1])**2
    PreResidual.append( [List[i][0], List[i][1], round(MeanY,4), PredictedY[i-1], round(SST,4), round(SSR,4), round(SSE,4)] )

#Re-Initialize
SST = 0
SSR = 0
SSE = 0

#Calculate sums of ... SST, SSR & SSE to actually extract the SST/R/E
for i in range(1,len(PreResidual)):
    SST += PreResidual[i][4]
    SSR += PreResidual[i][5]
    SSE += PreResidual[i][6]

#Check SST = SSR + SSE
if round(SST,-1) == round(SSR + SSE,-1):
    pass
    #print("indeed")

# ANOVA ... stuff for now (df in spss)
SSTdf = len(List)-2    #total = n-1
SSEdf = len(List[0])-2   #(exclude ID, 1st col and dependent variable Y)
SSRdf = SSTdf - SSEdf

# (Mean square in spss) = SSX/SSYdf (for SSE use SSRdf and for SSR use SSEdf)
MeanSquareSST = "Doesnt exist"    # doesnt exist
MeanSquareSSE = 0
MeanSquareSSR = 0
for i in range(1,len(PreResidual)):
    MeanSquareSSE += PreResidual[i][6] / SSRdf
    MeanSquareSSR += PreResidual[i][5] / SSEdf

# F ( in anova still)
F = MeanSquareSSR/MeanSquareSSE

##### R (Residual), R-square, Adjusted R square  ( = SSR/SST == .0963)
R = (SSR/SST)**0.5
RSquared = SSR/SST
AdjustedRSquared = 1 - ( (SSE/SSRdf)/(SST/SSTdf) )
#Adjusted RT-square test
if round(AdjustedRSquared,4) ==    round(  1 - ( (1 - RSquared)*(SSTdf - 1)/(SSTdf - SSEdf -1) )  ,4):
    pass
    #print("Indeed it is ")

### Sig = P(F >= Fobserved) // the probability/chance for f being greater than the one we have
# SEoE = Std. error of the estimate (in SPSS) // = square root of the Mean Square Residual (or Error).
SEoE = MeanSquareSSE**0.5


###### Unstandardtized Coefficients (b already  calculated above)
# Unstandardtized Coefficients , Std. Error 

B = []
Barray = np.array(Bmatrix)
for i in Barray:
    B.append(i[0])

# Transpose (X**T) the list for easier calculations using numpy
ListT = np.matrix.transpose(np.array(List))    # not utilized

# Create a list for the extra data (sum of each column)
ListSums= [List[0][1:],[]]

for j in range(1,len(List[0])):
    Count=0
    for i in range(1,len(List)):
        Count += List[i][j]
    ListSums[1].append(Count)

# Create a list for the extra data (mean for each column)
ListMeans = [List[0][1:],[]]

for j in range(1,len(List[0])):
    Count=0
    for i in range(1,len(List)):
        Count += List[i][j]
    ListMeans[1].append(Count/ (len(List) -1 ) )

# Calculate residuals for all dependent variables (X)
ListResiduals = List
for j in range(1,len(List[0])):
    for i in range(1,len(List)):
        ListResiduals[i][j] = List[i][j] - ListMeans[1][j-1]

# Calculate squared residuals for all dependent variables (X)
ListResidualsSquared = List
for j in range(1,len(List[0])):
    for i in range(1,len(List)):
        ListResidualsSquared[i][j] = (List[i][j] - ListMeans[1][j-1])**2

# Create a list for the extra data (sum of each column)
ListResidualsSquaredSums = [List[0][1:],[]]

for j in range(1,len(ListResidualsSquared[0])):
    Count=0
    for i in range(1,len(ListResidualsSquared)):
        Count += ListResidualsSquared[i][j]
    ListResidualsSquaredSums[1].append(Count)

##### cancel above method

#convert XtXinv into a python list
XtXinvList = [List[0][1:],[]]
for i in np.diag(XtXinv):
    XtXinvList[1].append(float(i))

#SSRdf

#Errors E = X*B - Y
Ematrix = Xmatrix*Bmatrix - Ymatrix

#DOF (degrees of freedom) = f (313 - 18 = 295)
DOF = len(Xmatrix)-len(Bmatrix)

#Standard Error
#Sigma0 = ((float(np.dot((np.matrix.transpose(Ematrix)),Ematrix)))/f)**0.5
Sigma0 = SEoE

#pinakas Symmetavlitotitas agnoston Edd(matrix)

Variancematrix = (Sigma0**2)*XtXinv

#typika sfalmata ton parametron
VarianceDiagonalmatrix = Variancematrix.diagonal()

VarianceDiagSquared = []
for i in range(len(Bmatrix)):
    VarianceDiagSquared.append((VarianceDiagonalmatrix[0,i])**2)
VarianceDiagSquaredmatrix = np.matrix(VarianceDiagSquared)

# convert into a list
VarianceDiagonal = [List[0][1:]]
for i in np.matrix.tolist(VarianceDiagonalmatrix):
    VarianceDiagonal.append(i)

# Coefficient Errors
CoefErrors = VarianceDiagonal
CoefErrors[1] = [float(i**0.5) for i in CoefErrors[1]]

# Standardized/beta Coefficients
# Standard Deviation
ListStandardDeviations = [ListResidualsSquaredSums[0],[]]
for i in ListResidualsSquaredSums[1]:
    ListStandardDeviations[1].append((i/(Rows))**0.5)
### Cancelled mathod ... quite possibly wrong :(

# Create a copy of the B list but insert float values (not numpy.float64)
Bcopy = []
for i in range(len(B)):
    Bcopy.append(float(B[i]))

# Coefficients T (t test?)
CoefTs=[[],[]]
'''Cancelled, would change list: CoefErrors
#for i in CoefErrors:
#    CoefTs.append(i)

#for i in range(len(CoefTs[1])):
#    CoefTs[1][i] = Bcopy[i]/CoefErrors[1][i]
'''
for i in range(len(CoefErrors[0])):
    CoefTs[0].append( CoefErrors[0][i] ) 
    CoefTs[1].append(  Bcopy[i]/CoefErrors[1][i]  )

# Create a .txt file with teh results
Results = open("RegressionResults.txt","w")
Results.write("----------Results of Regression----------\n\n")
Results.write("Dependent variable (Y):\n" + List[0][1]+"\n")
Results.write("Regression Equation:\nY = "+str(round(Bcopy[0],3)))
for i in range(1,len(Bcopy)):
    if Bcopy[i]>0:
        Results.write(" + "+str(round(Bcopy[i],3))+"*X"+str(i))
    elif Bcopy[i]<0:
        Results.write(" - "+str(round(abs(Bcopy[i]),3))+"*X"+str(i))
Results.write("\n")

Results.write("\n----------Model Summary----------\n")
Results.write("R\tR Square\t Adj. R Square\t Std. Error of Estimate\n")
Results.write(str(round(R,4))+"\t"+str(round(RSquared,4))+"\t\t\t"+str(round(AdjustedRSquared,4))+"\t\t"+str(round(SEoE,4))+"\n")

Results.write("\n----------ANOVA----------\n")
Results.write("\t\tSum of Squares\tdf\tMean Square\tF\n")
Results.write("Regression\t\t"+str(round(SSR,3))+"\t"+str(round(SSEdf,0))+"\t"+str(round(MeanSquareSSR,3))+"\t\t"+str(round(F,3))+"\n"  )
Results.write("Residual\t\t"+str(round(SSE,3))+"\t"+str(round(SSRdf,0))+"\t"+str(round(MeanSquareSSE,3))+"\n"  )
Results.write("total\t\t\t"+str(round(SST,3))+"\t"+str(round(SSTdf,0))+"\n"  )

Results.write("\n----------Coefficients----------\n")
Results.write("Unstandardized B\tCoefficients Std. Error\t\tt\n")
Results.write(str(round(Bcopy[0],3))+"\t\t\t"+str(round(CoefErrors[1][0],3))+"\t\t\t\t"+str(round(CoefTs[1][0],3))+"\t|Constant\n")
for i in range(1,len(B)):
    Results.write(str(round(Bcopy[i],3))+"\t\t\t"+str(round(CoefErrors[1][i],3))+"\t\t\t\t"+str(round(CoefTs[1][i],3))+"\t|"+List[0][i+1]+"\n")

Results.write("\n\nEND")
Results.close()


# Create an html file
html = open("RegressionReport.html","w")
html.write("<html>\n")
html.write("<title>Results of Regression</title>\n")
html.write("<link rel=\"icon\" type=\"image/x-icon\" href=\"https://www.uniwa.gr/wp-content/uploads/2018/11/logo-pada.png\">\n")
html.write("<style>table, th, td {  border:1px solid black;}</style>\n")
html.write("<style>th, td {text-align: center; }</style>\n")
html.write("<style>* {text-align: center; }</style>\n")
html.write("<style>table {margin: auto;} </style>\n")

html.write("<body>\n")
html.write("<h2>Dependent variale (Y): <strong>" + List[0][1]+"</strong></h2>\n")
html.write("<h2>\n")
html.write("Regression Equation:\n")
html.write("</h2>\n")
html.write("<h3>\n")
html.write("<pre>")
html.write("\nY = "+str(round(Bcopy[0],3)) )
for i in range(1,len(Bcopy)):
    if Bcopy[i]>0:
        html.write(" + "+str(round(Bcopy[i],3))+"*X"+str(i))
    elif Bcopy[i]<0:
        html.write(" - "+str(round(abs(Bcopy[i]),3))+"*X"+str(i))
html.write("\n")
html.write("</pre>\n")
html.write("</h3>\n")

'''
<table>
  <tr>
    <th>Company</th>
    <th>Contact</th>
    <th>Country</th>
  </tr>
  <tr>
    <td>Alfreds Futterkiste</td>
    <td>Maria Anders</td>
    <td>Germany</td>
  </tr>
</table>
'''

html.write("\n<h1>\nModel Summary\n</h1>")
html.write("\n<table style=\"width:40%\">\n")
html.write("\n<tr>")
html.write("\n<th>R</th>\n<th>R Square</th>\n<th>Adj. R Square</th>\n<th>Std. Error of Estimate</th>\n")
html.write("\n</tr>")
html.write("\n<tr>")
html.write("\n<td>"+str(round(R,4))+"</td>"   +"\n<td>"+str(round(RSquared,4))+"</td>"+   "\n<td>"+str(round(AdjustedRSquared,4))+"\n</td>"+   "\n<td>"+str(round(SEoE,4))+"</td>")
html.write("\n</tr>")
html.write("\n</table>\n")


html.write("\n<h1>ANOVA</h1>\n")
html.write("\n<table style=\"width:40%\">\n")
html.write("\n<tr>")
html.write("\n<th></th>\n<th>Sum of Squares</th>\n<th>   df   </th>\n<th>Mean Square</th>\n<th>   F   </th>")
html.write("\n</tr>")
html.write("\n<tr>")
html.write("\n<td>Regression</td>"+"\n<td>"+str(round(SSR,3))+"</td>"+"\n<td>"+str(round(SSEdf,0))+"</td>"+"\n<td>"+str(round(MeanSquareSSR,3))+"</td>"+"\n<td>"+str(round(F,3))+"</td>"+"\n"  )
html.write("\n</tr>")
html.write("\n<tr>")
html.write("\n<td>Residual</td>"+"\n<td>"+str(round(SSE,3))+"</td>"+"\n<td>"+str(round(SSRdf,0))+"</td>"+"\n<td>"+str(round(MeanSquareSSE,3))+"</td>"+"\n"  )
html.write("\n</tr>")
html.write("\n<tr>")
html.write("\n<td>total</td>"+"\n<td>"+str(round(SST,3))+"</td>"+"\n<td>"+str(round(SSTdf,0))+"</td>"+"\n"  )
html.write("\n</tr>")
html.write("\n</table>\n")


html.write("\n<h1>Coefficients</h1>\n")
html.write("\n<table style=\"width:40%\">\n")
html.write("\n<tr>")
html.write("\n<th></th>\n<th>Unstandardized Coefficients B</th>\n<th>Unstandardized Coefficients Std. Error</th>\n<th>t</th>\n")
html.write("\n</tr>")
html.write("\n<tr>")
html.write("\n<td>Constant</td>" + "<td>"+str(round(Bcopy[0],3))+"</td>" + "<td>"+str(round(CoefErrors[1][0],3))+"</td>" + "<td>"+str(round(CoefTs[1][0],3))+"</td>\n")
html.write("\n</tr>")

for i in range(1,len(B)):
    html.write("\n<tr>")
    html.write("\n<td>"+List[0][i+1]+"</td>"+ "\n<td>"+str(round(Bcopy[i],3))+"</td>" + "\n<td>"+str(round(CoefErrors[1][i],3))+"</td>" + "\n<td>"+str(round(CoefTs[1][i],3))+"</td>\n")
    html.write("\n</tr>")
html.write("\n</table>\n")

html.write("\n<br>\n<br>\n<br>")
html.write("\n</body>")
html.write("\n</html>")

html.close()




print("------------------END------------------")




