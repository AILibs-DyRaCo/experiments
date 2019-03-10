package dyadranking.sql;

import jaicore.basic.SQLAdapter;

public class SQLUtils {

	public static SQLAdapter sqlAdapterFromArgs(String[] args) {
		if (args.length < 4) {
			throw new IllegalArgumentException("Cannot find db credentials!");
		}
		String user = args[0];
		String password = args[1];
		String db_host = args[2];
		String db = args[3];
		return new SQLAdapter(db_host, user, password, db);
	}
}
