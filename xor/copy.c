#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define numTrainingSets 4
#define numInputs 2
#define numOutputs 1
#define numHiddenNodes 2
#define epochs 10000

// Activation function and its derivative
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }

// Initial weights and biases between 0.0 and 1.0
double init_weight(void) { return ((double)rand())/((double)RAND_MAX); }

//Shuffle the elements in an array
void shuffle(int *array, size_t n){
    //reset random number
    srand(time(NULL));
    if (n > 1){
        size_t i;
        for (i = 0; i<n; i++){
            size_t j = rand()/(RAND_MAX/(n));
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}//end shuffle for

int main()
{

    const double lr = 0.1f;

    double hiddenLayer[numHiddenNodes];//2
    double outputLayer[numOutputs];//1
    double hiddenLayerBias[numHiddenNodes];//2
    double outputLayerBias[numOutputs];//1
    double hiddenWeights[numInputs][numHiddenNodes];//2 2
    double outputWeights[numHiddenNodes][numOutputs];//2 1

    double training_inputs[numTrainingSets][numInputs]= { {0.0f,0.0f},{1.0f,0.0f},{0.0f,1.0f},{1.0f,1.0f} };//4 group input
    double training_outputs[numTrainingSets][numOutputs] = { {0.0f},{1.0f},{1.0f},{0.0f} };

    for (int i=0; i<numInputs; i++) {
        for (int j=0; j<numHiddenNodes; j++) {
            *(*(hiddenWeights+i)+j) = init_weight();
        }
    }//distribute weight to input


    for (int i=0; i<numHiddenNodes; i++) {
        *(hiddenLayerBias+i) = init_weight();//distribute bias to hidden layer node
        for (int j=0; j<numOutputs; j++) {
            *(*(outputWeights+i)+j) = init_weight();
        }
    }//distribute weight to hidden layer


    for (int i=0; i<numOutputs; i++) {
        *(outputLayerBias+i) = init_weight();
    }//distribute bias to output layer node

    

    // Iterate through the entire training for a number of epochs
    for (int n=0; n < epochs; n++) {

        // As per SGD, shuffle the order of the training set
        int trainingSetOrder[] = {0,1,2,3};
        shuffle(trainingSetOrder,numTrainingSets);

        // Cycle through each of the training set elements
        for (int x=0; x<numTrainingSets; x++) {
            int i = trainingSetOrder[x];

            // Forward pass

            for (int j=0; j<numHiddenNodes; j++) {// Compute hidden layer activation
                double activation=*(hiddenLayerBias+j);
                 for (int k=0; k<numInputs; k++) {
                    activation+=*(*(training_inputs+i)+k) * *(*(hiddenWeights+k)+j);
                }
                *(hiddenLayer+j) = sigmoid(activation);
            }

            for (int j=0; j<numOutputs; j++) {// Compute output layer activation
                double activation=*(outputLayerBias+j);
                for (int k=0; k<numHiddenNodes; k++) {
                    activation+=*(hiddenLayer+k)* *(*(outputWeights+k)+j);
                }
                *(outputLayer+j) = sigmoid(activation);
            }

            printf("Input: %f %f    Output: %f    Expected Output: %f  \n"
                   ,*(*(training_inputs+i)+0),*(*(training_inputs+i)+1),*(outputLayer),*(*(training_outputs+i)+0));

           // Backprop

            // Compute change in output weights
            double deltaOutput[numOutputs];
            for (int j=0; j<numOutputs; j++) {
                double errorOutput = (*(*(training_outputs+i)+j)- *(outputLayer+j));
                deltaOutput[j] = errorOutput*dSigmoid(*(outputLayer+j));
            }

            // Compute change in hidden weights
            double deltaHidden[numHiddenNodes];
            for (int j=0; j<numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k<numOutputs; k++) {
                    errorHidden+=deltaOutput[k]* *(*(outputWeights+j)+k);
                }
                deltaHidden[j] = errorHidden*dSigmoid(*(hiddenLayer+j));
            }

            // Apply change in output weights
            for (int j=0; j<numOutputs; j++) {
                *(outputLayerBias+j) += deltaOutput[j]*lr;
                for (int k=0; k<numHiddenNodes; k++) {
                    *(*(outputWeights+k)+j)+=(*(hiddenLayer+k))*deltaOutput[j]*lr;
                }
            }
            // Apply change in hidden weights
            for (int j=0; j<numHiddenNodes; j++) {
                *(hiddenLayerBias+j) += deltaHidden[j]*lr;
                for(int k=0; k<numInputs; k++) {
                    *(*(hiddenWeights+k)+j)+=*(*(training_inputs+i)+k)*deltaHidden[j]*lr;
                }
            }
        }
    }//end for iterate


    // Print weights
    printf("Final Hidden Weights\n[ ");
    for (int j=0; j<numHiddenNodes; j++) {
        printf("[ ");
        for(int k=0; k<numInputs; k++) {
            printf("%f  " , *(*(hiddenWeights+k)+j));
        }
        printf("] ");
    }
    printf("]\n");

    printf("Final Hidden Biases\n[ ");
    for (int j=0; j<numHiddenNodes; j++) {
        printf("%f  ",*(hiddenLayerBias+j));
    }
    printf("]\n");
    printf("Final Output Weights");
    for (int j=0; j<numOutputs; j++) {
        printf("[ ");
        for (int k=0; k<numHiddenNodes; k++) {
            printf("%f  ",*(*(outputWeights+k)+j));
        }
        printf("]\n");
    }
    printf("Final Output Biases\n[ ");
    for (int j=0; j<numOutputs; j++) {
        printf("%f  ",*(outputLayerBias+j));

    }
    printf("]\n");

    return 0;

}


