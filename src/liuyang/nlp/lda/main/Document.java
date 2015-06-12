package liuyang.nlp.lda.main;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import liuyang.nlp.lda.com.FileUtil;
import liuyang.nlp.lda.com.Stopwords;

public class Document {	
	private String docName;
	public ArrayList<String> words;
	int[] docWords;
	
	/**
	 * 单独读取一篇文档时用的构造函数
	 * @param docName
	 */
	public Document(String docName){
		//初始化
		this.docName = docName;	
		ArrayList<String> docLines = new ArrayList<String>();
		ArrayList<String> words = new ArrayList<String>();
		FileUtil.readLines(docName, docLines);
		for(String line : docLines){
			FileUtil.tokenizeAndLowerCase(line, words);
		}
		//移除stop word和 noise word
		for(int i = 0; i < words.size(); i++){
			if(Stopwords.isStopword(words.get(i)) || isNoiseWord(words.get(i))){
				words.remove(i);
				i--;
			}
		}
		this.words = words;
		//words.clear();
		
	}
	
	
	/**
	 * 在Documents初始化时用的构造函数
	 * @param docName
	 * @param termToIndexMap
	 * @param indexToTermMap
	 * @param termCountMap
	 */
	public Document(String docName, Map<String, Integer> termToIndexMap,
			ArrayList<String> indexToTermMap, Map<String, Integer> termCountMap){
		this.docName = docName;
		//Read file and initialize word index array
		ArrayList<String> docLines = new ArrayList<String>();
		ArrayList<String> words = new ArrayList<String>();
		FileUtil.readLines(docName, docLines);
		for(String line : docLines){
			FileUtil.tokenizeAndLowerCase(line, words);
		}
		//Remove stop words and noise words
		for(int i = 0; i < words.size(); i++){
			if(Stopwords.isStopword(words.get(i)) || isNoiseWord(words.get(i))){
				words.remove(i);
				i--;
			}
		}
		//Transfer word to index
		this.docWords = new int[words.size()];
		for(int i = 0; i < words.size(); i++){
			String word = words.get(i);
			if(!termToIndexMap.containsKey(word)){
				int newIndex = termToIndexMap.size();
				termToIndexMap.put(word, newIndex);
				indexToTermMap.add(word);
				termCountMap.put(word, new Integer(1));
				docWords[i] = newIndex;
			} else {
				docWords[i] = termToIndexMap.get(word);
				termCountMap.put(word, termCountMap.get(word) + 1);
			}
		}
		words.clear();
	}
	
	/*
	public boolean isNoiseWord(String string) {
		// TODO Auto-generated method stub
		string = string.toLowerCase().trim();
		Pattern MY_PATTERN = Pattern.compile(".*[a-zA-Z]+.*");
		Matcher m = MY_PATTERN.matcher(string);
		// filter @xxx and URL
		if(string.matches(".*www\\..*") || string.matches(".*\\.com.*") || 
				string.matches(".*http:.*") )
			return true;
		if (!m.matches()) {
			return true;
		} else
			return false;
	}
	*/
	public boolean isNoiseWord(String string) {
		// TODO Auto-generated method stub
		// filter @xxx and URL
		if(string.matches(".*www\\..*") || string.matches(".*\\.com.*") || 
				string.matches(".*http:.*") ){
			return true;
		}

		Pattern MY_PATTERN = Pattern.compile("[a-zA-Z\u4e00-\u9fa5]+");
		Matcher m = MY_PATTERN.matcher(string);
		if (!m.matches()) {
			return true;
		} else {
			return false;
		}
	}
	
	
	
}