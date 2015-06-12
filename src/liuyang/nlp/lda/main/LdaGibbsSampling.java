package liuyang.nlp.lda.main;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

import liuyang.nlp.lda.com.FileUtil;
import liuyang.nlp.lda.conf.ConstantConfig;
import liuyang.nlp.lda.conf.PathConfig;

/**Liu Yang's implementation of Gibbs Sampling of LDA
 * @author yangliu
 * @blog http://blog.csdn.net/yangliuy
 * @mail yangliuyx@gmail.com
 */

public class LdaGibbsSampling {
	
	public static class modelparameters {
		float alpha = 0.5f; //usual value is 50 / K
		float beta = 0.1f;//usual value is 0.1
		int topicNum = 100;
		int iteration = 100;
		int saveStep = 10;
		int beginSaveIters = 50;
	}
	
	/**Get parameters from configuring file. If the 
	 * configuring file has value in it, use the value.
	 * Else the default value in program will be used
	 * @param ldaparameters
	 * @param parameterFile
	 * @return void
	 */
	private static void getParametersFromFile(modelparameters ldaparameters,
			String parameterFile) {
		// TODO Auto-generated method stub
		ArrayList<String> paramLines = new ArrayList<String>();
		FileUtil.readLines(parameterFile, paramLines);
		for(String line : paramLines){
			String[] lineParts = line.split("\t");
			switch(parameters.valueOf(lineParts[0])){
			case alpha:
				ldaparameters.alpha = Float.valueOf(lineParts[1]);
				break;
			case beta:
				ldaparameters.beta = Float.valueOf(lineParts[1]);
				break;
			case topicNum:
				ldaparameters.topicNum = Integer.valueOf(lineParts[1]);
				break;
			case iteration:
				ldaparameters.iteration = Integer.valueOf(lineParts[1]);
				break;
			case saveStep:
				ldaparameters.saveStep = Integer.valueOf(lineParts[1]);
				break;
			case beginSaveIters:
				ldaparameters.beginSaveIters = Integer.valueOf(lineParts[1]);
				break;
			}
		}
	}
	
	public enum parameters{
		alpha, beta, topicNum, iteration, saveStep, beginSaveIters;
	}
	
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void train() throws IOException{
		//
		// 参数的路径
		String originalDocsPath = PathConfig.ldaDocsPath;
		String resultPath = PathConfig.LdaResultsPath;
		String parameterFile= ConstantConfig.LDAPARAMETERFILE;
		
		modelparameters ldaparameters = new modelparameters();
		getParametersFromFile(ldaparameters, parameterFile);
		Documents docSet = new Documents();
		docSet.readDocs(originalDocsPath);
		//added by tokyo 
		System.out.println("0 save termToIndexMap to file ...");
		docSet.savetermToIndexMap();
		
		System.out.println("wordMap size " + docSet.termToIndexMap.size());
		FileUtil.mkdir(new File(resultPath));
		LdaModel model = new LdaModel(ldaparameters);
		System.out.println("1 Initialize the model ...");
		model.initializeModel(docSet);
		System.out.println("2 Learning and Saving the model ...");
		model.inferenceModel(docSet);
		System.out.println("3 Output the final model ...");
		model.saveIteratedModel(ldaparameters.iteration, docSet);
		System.out.println("Done!");
	}
	
	public static void inference(){

		Document newDoc = new Document(PathConfig.ldaNewDocPath );
		//读取phi与termToIndexMap
		Map<String,Integer> termToIndexMap = FileUtil.readMapfromFile();
		Double[][] phi = FileUtil.readMatrixFromFile();
		
		//初始化Model
		String parameter_file= ConstantConfig.LDAPARAMETERFILE;
		modelparameters lda_parameters = new modelparameters();
		getParametersFromFile(lda_parameters, parameter_file);
		LdaModel c_model = new LdaModel(lda_parameters);
		//开始推理
		double[] res = c_model.classify(newDoc, termToIndexMap, phi);
		for (int i = 0;i<res.length;++i){
			System.out.println("topic"+i+":"+res[i]);
		}
		
		
	}
	
	public static void main(String[] args) throws IOException  {
		inference();
		//train();
	}
}
