package liuyang.nlp.lda.main;

/**Class for Lda model
 * @author yangliu
 * @blog http://blog.csdn.net/yangliuy
 * @mail yangliuyx@gmail.com
 */
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import liuyang.nlp.lda.com.FileUtil;
import liuyang.nlp.lda.conf.PathConfig;

public class LdaModel {
	
	int [][] doc;//word index array
	int V, K, M;//vocabulary size, topic number, document number
	int [][] z;//topic label array
	float alpha; //doc-topic dirichlet prior parameter 
	float beta; //topic-word dirichlet prior parameter
	int [][] nmk;//given document m, count times of topic k. M*K
	int [][] nkt;//given topic k, count times of term t. K*V
	int [] nmkSum;//Sum for each row in nmk
	int [] nktSum;//Sum for each row in nkt
	double [][] phi;//Parameters for topic-word distribution K*V
	double [][] theta;//Parameters for doc-topic distribution M*K
	int iterations;//Times of iterations
	int saveStep;//The number of iterations between two saving
	int beginSaveIters;//Begin save model at this iteration
	
	public LdaModel(LdaGibbsSampling.modelparameters modelparam) {
		// TODO Auto-generated constructor stub
		alpha = modelparam.alpha;
		beta = modelparam.beta;
		iterations = modelparam.iteration;
		K = modelparam.topicNum;
		saveStep = modelparam.saveStep;
		beginSaveIters = modelparam.beginSaveIters;
	}

	public void initializeModel(Documents docSet) {
		// TODO Auto-generated method stub
		M = docSet.docs.size();
		V = docSet.termToIndexMap.size();
		nmk = new int [M][K];
		nkt = new int[K][V];
		nmkSum = new int[M];
		nktSum = new int[K];
		phi = new double[K][V];
		theta = new double[M][K];
		
		//initialize documents index array
		doc = new int[M][];
		for(int m = 0; m < M; m++){
			//Notice the limit of memory
			int N = docSet.docs.get(m).docWords.length;
			doc[m] = new int[N];
			for(int n = 0; n < N; n++){
				doc[m][n] = docSet.docs.get(m).docWords[n];
			}
		}
		
		//initialize topic lable z for each word
		z = new int[M][];
		for(int m = 0; m < M; m++){
			int N = docSet.docs.get(m).docWords.length;
			z[m] = new int[N];
			for(int n = 0; n < N; n++){
				int initTopic = (int)(Math.random() * K);// From 0 to K - 1
				z[m][n] = initTopic;
				//number of words in doc m assigned to topic initTopic add 1
				nmk[m][initTopic]++;
				//number of terms doc[m][n] assigned to topic initTopic add 1
				nkt[initTopic][doc[m][n]]++;
				// total number of words assigned to topic initTopic add 1
				nktSum[initTopic]++;
			}
			 // total number of words in document m is N
			nmkSum[m] = N;
		}
	}

	public void inferenceModel(Documents docSet) throws IOException {
		// TODO Auto-generated method stub
		if(iterations < saveStep + beginSaveIters){
			System.err.println("Error: the number of iterations should be larger than " + (saveStep + beginSaveIters));
			System.exit(0);
		}
		for(int i = 0; i < iterations; i++){
			System.out.println("Iteration " + i);
			if((i >= beginSaveIters) && (((i - beginSaveIters) % saveStep) == 0)){
				//Saving the model
				System.out.println("Saving model at iteration " + i +" ... ");
				//Firstly update parameters
				updateEstimatedParameters();
				//Secondly print model variables
				saveIteratedModel(i, docSet);
			}
			
			//Use Gibbs Sampling to update z[][]
			for(int m = 0; m < M; m++){
				int N = docSet.docs.get(m).docWords.length;
				for(int n = 0; n < N; n++){
					// Sample from p(z_i|z_-i, w)
					int newTopic = sampleTopicZ(m, n);
					z[m][n] = newTopic;
				}
			}
		}
	}
	
	

	/**
	 * Add the classify part of LDA
	 * @param doc 文档
	 * @param termToIndexMap 用于索引word序号
	 * @return doc的topic分布 :theta
	 */
	public double[] classify(Document doc, Map<String, Integer> termToIndexMap,Double[][] phi){
		
		ArrayList<String> oldWords = doc.words;
		ArrayList<String> newWords = new ArrayList<String>();
		Set<String> wordSet = new HashSet<String>();
		Map<Integer,Integer> transTable = new HashMap<Integer,Integer>();
	
		//将训练集合中没有的word排除，并通过集合去重
		for(String word : oldWords){
			if(termToIndexMap.containsKey(word)){
				newWords.add(word);
				wordSet.add(word);
			}
		}
		
		//定义变量
		int nWords = newWords.size();
		int d_V = wordSet.size();
	
		//d_theta为主题分布
		double[] d_theta = new double[K];
		//d_z记录每个word的主题
		int[] d_z = new int[nWords];
		//d_V表示不重复的word的个数
		
		//d_nk表示              topic1  topic2  ...  topicK
		//      doc  0.1     0.1     ...  0.1 
		int[] d_nk = new int[K];
		//d_nkSum表示d_nk中每行的sum
		int d_nkSum = 0;
		
		int[][] d_nkt = new int[K][d_V];
		int[] d_nktSum = new int[K];
		
		
		//d_doc表示每个词的序号
		int[] d_doc = new int[nWords];
		
		//给doc中的word标号
		int itemNum = 0;
		for(int i = 0; i < nWords; ++i){
			String word = newWords.get(i);
			int index = termToIndexMap.get(word);
			d_doc[i] = index;
			if (!transTable.containsKey(index)){
				transTable.put(index, itemNum++);
			}
		}
		
		//初始化topic
		for(int i = 0; i < nWords; ++i){
			int initTopic = (int)(Math.random() * K);
			d_z[i] = initTopic;
			//number of words in doc m assigned to topic initTopic add 1
			d_nk[initTopic]++;
			//number of terms doc[m][n] assigned to topic initTopic add 1
			int t = d_doc[i];
			int trans_t = transTable.get(t);
			d_nkt[initTopic][trans_t]++;
			// total number of words assigned to topic initTopic add 1
			d_nktSum[initTopic]++;
		}
		d_nkSum = nWords;
		
		//开始采样d_theta
		int d_iterations = 100;
		int sample_step = 5;
		for(int i = 0; i < d_iterations; i++){
			
			//以采样步长进行参数更新
			if(i % d_iterations == sample_step){
				for(int k = 0; k < K; k++){
					d_theta[k] = (d_nk[k] + alpha) / (d_nkSum + K * alpha);
				}
			}
			
			//重复进行吉布斯采样,更新d_z
			for(int n = 0; n < nWords; n++){
				// Sample from p(z_i|z_-i, w)
				//这里利用多态，再定义一个sampleTopicZ方法
				int newTopic = sampleTopicZ(n,d_z,d_doc,transTable,
						d_nk,d_nkSum,d_nkt,d_nktSum,phi);
				d_z[n] = newTopic;
			}
		}
		
		//归一化处理
		double d_thetaSum = 0.0;
		for (double x:d_theta){
			d_thetaSum += x;
		}
		for (int i=0;i< d_theta.length;++i){
			d_theta[i] = d_theta[i] / d_thetaSum;
		}
		return d_theta;
		
	}
	
	private void updateEstimatedParameters() {
		// TODO Auto-generated method stub
		for(int k = 0; k < K; k++){
			for(int t = 0; t < V; t++){
				phi[k][t] = (nkt[k][t] + beta) / (nktSum[k] + V * beta);
			}
		}
		
		for(int m = 0; m < M; m++){
			for(int k = 0; k < K; k++){
				theta[m][k] = (nmk[m][k] + alpha) / (nmkSum[m] + K * alpha);
			}
		}
	}

	
	/**
	 * 对新输入未分类文档进行采样
	 * @param n
	 * @param d_nk
	 * @param d_nkSum
	 * @param d_nkt
	 * @param d_nktSum
	 * @param phi
	 * @return
	 */
	private int sampleTopicZ(int n,int[] d_z,int[] d_doc,Map<Integer,Integer> transTable,
			int[] d_nk, Integer d_nkSum,
			int[][] d_nkt, int[] d_nktSum,
			Double[][] phi){
		
		int d_V = d_nkt[0].length;
		int oldTopic = d_z[n];
		int origin_t = d_doc[n];
		int t = transTable.get(origin_t);
		
		d_nk[oldTopic]--;
		d_nkt[oldTopic][t]--;
		d_nkSum--;
		d_nktSum[oldTopic]--;
		//计算 p(z_i = k|z_-i, w)
		double [] p = new double[K];
		for(int k = 0; k < K; k++){
			p[k] = (d_nkt[k][t] + beta) / (d_nktSum[k] + d_V * beta) * phi[k][t];
		}
		//利用赌轮法确定topic
		for(int k = 1; k < K; k++){
			p[k] += p[k - 1];
		}
		double u = Math.random() * p[K - 1]; //p[] is unnormalised
		int newTopic;
		for(newTopic = 0; newTopic < K; newTopic++){
			if(u < p[newTopic]){
				break;
			}
		}
		//更新这个文档第n个词的topic
		d_nk[newTopic]++;
		d_nkt[newTopic][t]++;
		d_nkSum++;
		d_nktSum[newTopic]++;
		return newTopic;
		
	}
	
	
	
	private int sampleTopicZ(int m, int n) {
		// TODO Auto-generated method stub
		// Sample from p(z_i|z_-i, w) using Gibbs upde rule
		
		//Remove topic label for w_{m,n}
		int oldTopic = z[m][n];
		nmk[m][oldTopic]--;
		nkt[oldTopic][doc[m][n]]--;
		nmkSum[m]--;
		nktSum[oldTopic]--;
		
		//Compute p(z_i = k|z_-i, w)
		double [] p = new double[K];
		for(int k = 0; k < K; k++){
			p[k] = (nkt[k][doc[m][n]] + beta) / (nktSum[k] + V * beta) * (nmk[m][k] + alpha) / (nmkSum[m] + K * alpha);
		}
		
		//Sample a new topic label for w_{m, n} like roulette
		//Compute cumulated probability for p
		for(int k = 1; k < K; k++){
			p[k] += p[k - 1];
		}
		double u = Math.random() * p[K - 1]; //p[] is unnormalised
		int newTopic;
		for(newTopic = 0; newTopic < K; newTopic++){
			if(u < p[newTopic]){
				break;
			}
		}
		
		//Add new topic label for w_{m, n}
		nmk[m][newTopic]++;
		nkt[newTopic][doc[m][n]]++;
		nmkSum[m]++;
		nktSum[newTopic]++;
		return newTopic;
	}

	public void saveIteratedModel(int iters, Documents docSet) throws IOException {
		// TODO Auto-generated method stub
		//lda.params lda.phi lda.theta lda.tassign lda.twords
		//lda.params
		String resPath = PathConfig.LdaResultsPath;
		String modelName = "lda_" + iters;
		ArrayList<String> lines = new ArrayList<String>();
		lines.add("alpha = " + alpha);
		lines.add("beta = " + beta);
		lines.add("topicNum = " + K);
		lines.add("docNum = " + M);
		lines.add("termNum = " + V);
		lines.add("iterations = " + iterations);
		lines.add("saveStep = " + saveStep);
		lines.add("beginSaveIters = " + beginSaveIters);
		FileUtil.writeLines(resPath + modelName + ".params", lines);
		
		//lda.phi K*V
		BufferedWriter writer = new BufferedWriter(new FileWriter(resPath + modelName + ".phi"));		
		for (int i = 0; i < K; i++){
			for (int j = 0; j < V; j++){
				writer.write(phi[i][j] + "\t");
			}
			writer.write("\n");
		}
		writer.close();
		
		//lda.theta M*K
		writer = new BufferedWriter(new FileWriter(resPath + modelName + ".theta"));
		for(int i = 0; i < M; i++){
			for(int j = 0; j < K; j++){
				writer.write(theta[i][j] + "\t");
			}
			writer.write("\n");
		}
		writer.close();
		
		//lda.tassign
		writer = new BufferedWriter(new FileWriter(resPath + modelName + ".tassign"));
		for(int m = 0; m < M; m++){
			for(int n = 0; n < doc[m].length; n++){
				writer.write(doc[m][n] + ":" + z[m][n] + "\t");
			}
			writer.write("\n");
		}
		writer.close();
		
		//lda.twords phi[][] K*V
		writer = new BufferedWriter(new FileWriter(resPath + modelName + ".twords"));
		int topNum = 20; //Find the top 20 topic words in each topic
		for(int i = 0; i < K; i++){
			List<Integer> tWordsIndexArray = new ArrayList<Integer>(); 
			for(int j = 0; j < V; j++){
				tWordsIndexArray.add(new Integer(j));
			}
			Collections.sort(tWordsIndexArray, new LdaModel.TwordsComparable(phi[i]));
			writer.write("topic " + i + "\t:\t");
			for(int t = 0; t < topNum; t++){
				writer.write(docSet.indexToTermMap.get(tWordsIndexArray.get(t)) + " " + phi[i][tWordsIndexArray.get(t)] + "\t");
			}
			writer.write("\n");
		}
		writer.close();
	}
	
	public class TwordsComparable implements Comparator<Integer> {
		
		public double [] sortProb; // Store probability of each word in topic k
		
		public TwordsComparable (double[] sortProb){
			this.sortProb = sortProb;
		}

		@Override
		public int compare(Integer o1, Integer o2) {
			// TODO Auto-generated method stub
			//Sort topic word index according to the probability of each word in topic k
			if(sortProb[o1] > sortProb[o2]) return -1;
			else if(sortProb[o1] < sortProb[o2]) return 1;
			else return 0;
		}
	}
}
