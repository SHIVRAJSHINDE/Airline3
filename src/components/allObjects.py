
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion

from src.components.eda import EDA
from src.components.encoding import encodingclass
from src.components.eda import SplitTheData
from src.components.data_transformation import DataTransformation

#from src.components.train import ModelTrainer
#from src.components.train import ModelTrainerConfig


if __name__ == "__main__":
    readFilefromCSVObj = DataIngestion()
    dataFile = readFilefromCSVObj.initiate_data_ingestion()

    edaObj = EDA()

    dataWithEda = edaObj.edaOfTrainData(dataFile)

    encodingObj1 = encodingclass(dataWithEda)
    #encoingObj2 = encodingclass(test_data)

    trainTestSplit = SplitTheData()
    train_data_path,test_data_path = trainTestSplit.trainTestSplit()

    dataTransformation = DataTransformation()
    dataTransformation.initiate_data_transformation(train_data_path,test_data_path)

    #dataWithEdaAndEncoding = encodingObj1.trainDataencoding()

    #data_test = encoingObj2.trainDataencoding()

    #transformData = transformClass()
    #train_arr, test_arr = transformData.initiate_data_transformation(data_train,data_test)

    #modeltrainer = ModelTrainer()
    #modeltrainer.initiate_model_trainer(train_arr, test_arr)




