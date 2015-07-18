/**
 * 
 */
package com.xiang.topicmodel.lda;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;



import com.xiang.topicmodel.lda.FileUtil;

import com.xiang.topicmodel.framework.Trainer;

/**
 * @author rivercrab3
 *
 */
public class LdaTrainer extends Trainer {
	
	String paramFileName;
	LdaDataProc dataproc;
	
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
	
	
	public LdaTrainer(String paramFileName, LdaDataProc ldp) throws IOException{
		this.paramFileName = paramFileName;
		this.dataproc = ldp;
		Properties prop = new Properties();
		File pFile = new File(paramFileName);
		FileInputStream pfin = new FileInputStream(pFile);
		prop.load(pfin);
		this.alpha = Float.parseFloat(prop.getProperty(("alpha")));
		this.beta = Float.parseFloat(prop.getProperty(("beta")));
		this.iterations = Integer.parseInt(prop.getProperty(("iterations")));
		this.K = Integer.parseInt(prop.getProperty(("K")));
		this.saveStep = Integer.parseInt(prop.getProperty(("saveStep")));
		this.beginSaveIters = Integer.parseInt(prop.getProperty(("beginSaveIters")));
	}
	
	
	
	public void train() throws IOException{
		Properties prop = new Properties();
		File pFile = new File(this.paramFileName);
		FileInputStream pfin = new FileInputStream(pFile);
		prop.load(pfin);
		String originalDocsPath = prop.getProperty(("ldaDocsPath"));
		String resultPath = prop.getProperty(("ldaResultsPath"));

		System.out.println("0 save termToIndexMap to file ...");
		this.dataproc.savetermToIndexMap(resultPath);
		
		System.out.println("wordMap size " + this.dataproc.termToIndexMap.size());
		FileUtil.mkdir(new File(resultPath));
		System.out.println("1 Initialize the model ...");
		this.initialize(this.dataproc);
		System.out.println("2 Learning and Saving the model ...");
		this.inference(this.dataproc);
		System.out.println("3 Output the final model ...");
		this.saveIteratedModel(this.iterations,this.paramFileName,this.dataproc);
		System.out.println("Done!");		
	}
	
	
	private void initialize(LdaDataProc ldp) {
		// TODO Auto-generated method stub
		M = ldp.docs.size();
		V = ldp.termToIndexMap.size();
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
			ArrayList<String> mth_doc = ldp.docs.get(m);
			int N = mth_doc.size();
			doc[m] = new int[N];
			for(int n = 0; n < N; n++){
				String nth_word = mth_doc.get(n);
				Integer indx = ldp.termToIndexMap.get(nth_word);
				doc[m][n] = indx;
			}
		}
		
		//initialize topic lable z for each word
		z = new int[M][];
		for(int m = 0; m < M; m++){
			ArrayList<String> mth_doc = ldp.docs.get(m);
			int N = mth_doc.size();
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
	
	
	public void inference(LdaDataProc ldp) throws IOException {
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
				saveIteratedModel(i,this.paramFileName, ldp);
			}
			
			//Use Gibbs Sampling to update z[][]
			for(int m = 0; m < M; m++){
				ArrayList<String> mth_doc = ldp.docs.get(m);
				int N = mth_doc.size();
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
	 * @param doc �ĵ�
	 * @param termToIndexMap ��������word���
	 * @return doc��topic�ֲ� :theta
	 */
	public double[] classify(LdaDataProc ldp , Map<String, Integer> termToIndexMap,Double[][] phi){
		ArrayList<ArrayList<String>> docs = ldp.docs;
		if(docs.size() > 1 || docs.size() < 1){
			return null;
		}
		ArrayList<String> words = docs.get(0);
		ArrayList<String> oldWords = words;
		ArrayList<String> newWords = new ArrayList<String>();
		Set<String> wordSet = new HashSet<String>();
		Map<Integer,Integer> transTable = new HashMap<Integer,Integer>();
	
		//��ѵ��������û�е�word�ų�����ͨ������ȥ��
		for(String word : oldWords){
			if(termToIndexMap.containsKey(word)){
				newWords.add(word);
				wordSet.add(word);
			}
		}
		
		//�������
		int nWords = newWords.size();
		int d_V = wordSet.size();
	
		//d_thetaΪ����ֲ�
		double[] d_theta = new double[K];
		//d_z��¼ÿ��word������
		int[] d_z = new int[nWords];
		//d_V��ʾ���ظ���word�ĸ���
		
		//d_nk��ʾ              topic1  topic2  ...  topicK
		//      doc  0.1     0.1     ...  0.1 
		int[] d_nk = new int[K];
		//d_nkSum��ʾd_nk��ÿ�е�sum
		int d_nkSum = 0;
		
		int[][] d_nkt = new int[K][d_V];
		int[] d_nktSum = new int[K];
		
		
		//d_doc��ʾÿ���ʵ����
		int[] d_doc = new int[nWords];
		
		//��doc�е�word���
		int itemNum = 0;
		for(int i = 0; i < nWords; ++i){
			String word = newWords.get(i);
			int index = termToIndexMap.get(word);
			d_doc[i] = index;
			if (!transTable.containsKey(index)){
				transTable.put(index, itemNum++);
			}
		}
		
		//��ʼ��topic
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
		
		//��ʼ����d_theta
		int d_iterations = 100;
		int sample_step = 5;
		for(int i = 0; i < d_iterations; i++){
			
			//�Բ����������в�������
			if(i % d_iterations == sample_step){
				for(int k = 0; k < K; k++){
					d_theta[k] = (d_nk[k] + alpha) / (d_nkSum + K * alpha);
				}
			}
			
			//�ظ����м���˹����,����d_z
			for(int n = 0; n < nWords; n++){
				// Sample from p(z_i|z_-i, w)
				//�������ö�̬���ٶ���һ��sampleTopicZ����
				int newTopic = sampleTopicZ(n,d_z,d_doc,transTable,
						d_nk,d_nkSum,d_nkt,d_nktSum,phi);
				d_z[n] = newTopic;
			}
		}
		
		//��һ������
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
	 * ��������δ�����ĵ����в���
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
		//���� p(z_i = k|z_-i, w)
		double [] p = new double[K];
		for(int k = 0; k < K; k++){
			p[k] = (d_nkt[k][t] + beta) / (d_nktSum[k] + d_V * beta) * phi[k][t];
		}
		//���ö��ַ�ȷ��topic
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
		//��������ĵ���n���ʵ�topic
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

	public void saveIteratedModel(int iters,String paramFileName ,LdaDataProc ldp) throws IOException {
		// TODO Auto-generated method stub
		//lda.params lda.phi lda.theta lda.tassign lda.twords
		//lda.params
		
		Properties prop = new Properties();
		File pFile = new File(paramFileName);
		FileInputStream pfin = new FileInputStream(pFile);
		prop.load(pfin);
		String resPath = prop.getProperty(("ldaResultsPath"));
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
			Collections.sort(tWordsIndexArray, new LdaTrainer.TwordsComparable(phi[i]));
			writer.write("topic " + i + "\t:\t");
			for(int t = 0; t < topNum; t++){
				
				writer.write(ldp.indexToTermMap.get(tWordsIndexArray.get(t)) + " " + phi[i][tWordsIndexArray.get(t)] + "\t");
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
