package liuyang.nlp.lda.main;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import liuyang.nlp.lda.com.FileUtil;
import liuyang.nlp.lda.conf.PathConfig;


/**Class for corpus which consists of M documents
 * @author yangliu
 * @blog http://blog.csdn.net/yangliuy
 * @mail yangliuyx@gmail.com
 */

public class Documents {

	ArrayList<Document> docs; 
	Map<String, Integer> termToIndexMap;
	ArrayList<String> indexToTermMap;
	Map<String,Integer> termCountMap;

	public Documents(){
		docs = new ArrayList<Document>();
		termToIndexMap = new HashMap<String, Integer>();
		indexToTermMap = new ArrayList<String>();
		termCountMap = new HashMap<String, Integer>();
	}

	public void readDocs(String docsPath){
		for(File docFile : new File(docsPath).listFiles()){
			Document doc = new Document(docFile.getAbsolutePath(), termToIndexMap, indexToTermMap, termCountMap);
			docs.add(doc);
		}
	}

	//added by tokyo
	/**
	 * ±£´ætermToInDexMap
	 */
	public void savetermToIndexMap(){
		String filePath = PathConfig.LdaResultsPath;
		String fileName = "termToIndexMap";
		ArrayList<String> lines = new ArrayList<String>();
		for (Entry<String, Integer> termToIndex : termToIndexMap.entrySet()) {
			String line = termToIndex.getKey()+":"+termToIndex.getValue();
			lines.add(line);
		}
		FileUtil.writeLines(filePath + fileName, lines);
	}


}
